import inspect
import io
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import benchmark
import benchmark_s3
import jobs
from main import app


class S3:
    def __init__(self, rows=None, sessions=("good-one",)):
        self.rows, self.sessions, self.puts, self.presigns, self.storage = rows or [], sessions, [], [], {}

    def list_objects_v2(self, **kwargs): return {"Contents": self.rows, "IsTruncated": False}
    def get_object(self, Bucket, Key):
        if Key in self.storage:
            data = self.storage[Key]
        elif Key.endswith("take_judge_dataset.jsonl"):
            data = b"".join((json.dumps({"session_id": session, "clip_id": f"old-{session}",
                "text": "hello product", "keep": True, "slot": "HOOK", "source": "good"}) + "\n").encode()
                for session in self.sessions)
        else:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"ContentLength": len(data), "Body": io.BytesIO(data)}
    def put_object(self, **kwargs): self.puts.append(kwargs); self.storage[kwargs["Key"]] = kwargs["Body"]
    def generate_presigned_url(self, *args, **kwargs): self.presigns.append(kwargs); return "signed"


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "private-test-bucket")


def request(mode="old_vs_new"):
    return {"dataset_key": "editdna/training/take_judge_dataset.jsonl",
            "source_prefixes": ["Editdna good videos/"], "mode": mode}


def test_allowlists_and_listing_filters(monkeypatch):
    with pytest.raises(ValueError): benchmark_s3.validate_input_prefix("private/")
    with pytest.raises(ValueError): benchmark_s3.validate_dataset_key("editdna/training/../secret.jsonl")
    monkeypatch.setattr(benchmark_s3, "MIN_OBJECT_BYTES", 100)
    s3 = S3([{"Key": "Editdna good videos/", "Size": 0}, {"Key": "Editdna good videos/._x.mp4", "Size": 500},
             {"Key": "Editdna good videos/tiny.mp4", "Size": 2}, {"Key": "Editdna good videos/x.txt", "Size": 500},
             {"Key": "Editdna good videos/ok.mp4", "Size": 500}])
    found, stats = benchmark_s3.list_objects_inventory(s3, "Editdna good videos/")
    assert found == [{"key": "Editdna good videos/ok.mp4", "size": 500}]
    assert stats == {"filtered_s3_objects": 4, "eligible_s3_videos": 1}


def test_configured_prefixes_remain_closed_and_validated(monkeypatch):
    monkeypatch.setenv("BENCHMARK_GOOD_VIDEOS_PREFIX", "approved/good/")
    assert benchmark_s3.validate_input_prefix("approved/good/") == "approved/good/"
    with pytest.raises(ValueError): benchmark_s3.validate_input_prefix("Editdna good videos/")
    monkeypatch.setenv("BENCHMARK_GOOD_VIDEOS_PREFIX", "../escape/")
    with pytest.raises(ValueError): benchmark_s3.configured_input_prefixes()
    monkeypatch.setenv("BENCHMARK_GOOD_VIDEOS_PREFIX", "missing-slash")
    with pytest.raises(ValueError): benchmark_s3.configured_input_prefixes()


def test_exact_matching_ignores_generic_source_and_reports_uncertain():
    grouped = {"good-one": [{"source": "good"}], "missing": [{"source": "bloopers"}]}
    objects = [{"key": "Editdna good videos/good-one.mp4"}, {"key": "Editdna good videos/good-extra.mp4"}]
    resolved, unresolved = benchmark.resolve_sources(grouped, objects)
    assert resolved["good-one"]["key"].endswith("good-one.mp4")
    assert unresolved["missing"] == {"classification": "missing_source", "candidates": []}


def test_inventory_checkpoint_resume_progress_and_metrics():
    rows = [{"Key": "Editdna good videos/good-one.mp4", "Size": 2000},
            {"Key": "Editdna good videos/good-two.mp4", "Size": 2000}]
    s3 = S3(rows, sessions=("good-one", "good-two")); progress = []
    first = benchmark.run_benchmark("job", request("inventory_only"), s3=s3,
                                    pipeline=lambda *args: pytest.fail("pipeline invoked"), progress=progress.append)
    inventory = json.loads(s3.storage["editdna/benchmarks/job/inventory.json"])
    assert inventory[0] == {"session_id": "good-one", "resolution_status": "resolved",
                            "resolved_s3_key": "Editdna good videos/good-one.mp4",
                            "candidate_s3_keys": ["Editdna good videos/good-one.mp4"], "historical_clip_count": 1}
    assert first["summary"]["total_historical_sessions"] == first["summary"]["resolved_sessions"] == 2
    assert progress[-1]["processed_sessions"] == 2
    assert first["summary"]["preflight"]["benchmark_write"]["successful"] is True
    assert first["summary"]["preflight"]["benchmark_read_back"]["successful"] is True
    assert first["summary"]["preflight"]["configured_prefixes"] == [{
        "prefix": "Editdna good videos/", "filtered_s3_objects": 0, "eligible_s3_videos": 2}]
    calls = []
    benchmark.run_benchmark("job", request("inventory_only"), s3=s3, pipeline=lambda *args: calls.append(args))
    assert calls == []  # checkpoint automatically prevents reruns
    checkpoint = json.loads(s3.storage["editdna/benchmarks/job/checkpoint.json"])
    assert set(checkpoint["session_result_keys"]) == {"good-one", "good-two"}


def test_retryable_failure_resumes_then_succeeds_without_reprocessing(monkeypatch):
    monkeypatch.setenv("BENCHMARK_SESSION_MAX_ATTEMPTS", "3")
    rows = [{"Key": "Editdna good videos/good-one.mp4", "Size": 2000},
            {"Key": "Editdna good videos/good-two.mp4", "Size": 2000}]
    s3 = S3(rows, sessions=("good-one", "good-two")); calls = []
    def pipeline(session, *_):
        calls.append(session)
        if session == "good-one" and calls.count("good-one") == 1: raise RuntimeError("safe failure")
        return {"clips": [{"id": "new", "text": "hello product", "slot": "BENEFITS", "meta": {"keep": False}}],
                "provider_usage_instrumentation": {"available": False, "reason": "not instrumented"},
                "estimated_cost_instrumentation": {"available": False, "reason": "not instrumented"}}
    with pytest.raises(benchmark.RetryableBenchmarkSessionsError):
        benchmark.run_benchmark("job", request(), s3=s3, pipeline=pipeline)
    assert calls == ["good-one", "good-two"]
    checkpoint = json.loads(s3.storage["editdna/benchmarks/job/checkpoint.json"])
    assert checkpoint["completed_sessions"] == ["good-two"] and checkpoint["attempts_by_session"]["good-one"] == 1
    result = benchmark.run_benchmark("job", request(), s3=s3, pipeline=pipeline)
    assert calls == ["good-one", "good-two", "good-one"]
    summary = result["summary"]
    assert summary["failed_sessions"] == 0
    assert summary["per_slot_changes"] == {"HOOK->BENEFITS": 2}
    assert summary["provider_usage_instrumentation"]["available"] is False


def test_failure_becomes_terminal_after_attempt_limit(monkeypatch):
    monkeypatch.setenv("BENCHMARK_SESSION_MAX_ATTEMPTS", "2")
    rows = [{"Key": "Editdna good videos/good-one.mp4", "Size": 2000}]; s3 = S3(rows)
    failing = lambda *_: (_ for _ in ()).throw(RuntimeError("safe failure"))
    with pytest.raises(benchmark.RetryableBenchmarkSessionsError):
        benchmark.run_benchmark("terminal", request(), s3=s3, pipeline=failing)
    result = benchmark.run_benchmark("terminal", request(), s3=s3, pipeline=failing)
    checkpoint = json.loads(s3.storage["editdna/benchmarks/terminal/checkpoint.json"])
    assert checkpoint["attempts_by_session"] == {"good-one": 2}
    assert checkpoint["terminal_failed_sessions"] == checkpoint["completed_sessions"] == ["good-one"]
    session_key = checkpoint["session_result_keys"]["good-one"]
    assert len(json.loads(s3.storage[session_key])["errors"]) == 2
    assert result["summary"]["failed_sessions"] == 1


def test_take_judge_result_usage_fallback_and_winner_change():
    old = {"session_id": "s", "clip_id": "old", "text": "same", "slot": "HOOK", "keep": True,
           "take_judge_verdict": "LOSER"}
    with_judge = benchmark.make_result(old, {"id": "new", "text": "same", "slot": "HOOK",
        "meta": {"keep": True, "take_judge_score": .91, "take_judge_verdict": "WINNER"}}, "text", .9)
    fallback = benchmark.make_result(old, {"id": "new2", "text": "same", "slot": "HOOK",
        "meta": {"keep": True}}, "text", .9)
    assert with_judge["take_judge_v2"] == {"score": .91, "verdict": "WINNER"}
    assert with_judge["winner_selection_changed"] is True
    summary = benchmark.build_summary([old, old], ["s"], {"s": {}}, {}, [], 0,
        [with_judge, fallback], [], [], [], 0)
    assert summary["take_judge_v2_usage"] == 1
    assert summary["take_judge_v2_fallbacks"] == 1
    assert json.loads(benchmark.jsonl([with_judge]))["take_judge_v2"] == {"score": .91, "verdict": "WINNER"}


@pytest.mark.parametrize(("old_verdict", "new_verdict", "changed", "evaluation"), [
    ("WINNER", "LOSER", True, "evaluated"), ("WINNER", None, False, "not_evaluated"),
    (None, "WINNER", False, "newly_evaluated"), (None, None, False, "not_evaluated")])
def test_winner_change_requires_both_verdicts(old_verdict, new_verdict, changed, evaluation):
    result = benchmark.make_result({"session_id": "s", "text": "same", "take_judge_verdict": old_verdict},
        {"id": "n", "text": "same", "meta": {"take_judge_verdict": new_verdict}}, "text", 1)
    assert result["winner_selection_changed"] is changed
    assert result["take_judge_evaluation"] == evaluation


def test_take_judge_execution_statuses_are_request_scoped(monkeypatch):
    from worker import pipeline
    monkeypatch.setattr(pipeline, "TAKE_JUDGE_ENABLED", False)
    status = {}
    pipeline.run_take_judge([], "/tmp", "input", execution_status=status)
    assert status["status"] == "not_requested"
    candidates = [{"id": value, "meta": {"keep": True}} for value in ("a", "b")]
    monkeypatch.setattr(pipeline, "find_sibling_groups", lambda clips: [candidates])
    monkeypatch.setattr(pipeline, "is_openai_available", lambda: False)
    pipeline.run_take_judge(candidates, "/tmp", "input", force_enabled=True, execution_status=status)
    assert status["status"] == "provider_unavailable"
    assert all(item["meta"]["take_judge_execution_status"] == "provider_unavailable" for item in candidates)
    monkeypatch.setattr(pipeline, "is_openai_available", lambda: True)
    monkeypatch.setattr(pipeline, "find_sibling_groups", lambda clips: [])
    pipeline.run_take_judge([], "/tmp", "input", force_enabled=True, execution_status=status)
    assert status["status"] == "no_sibling_group"


def test_take_judge_missing_evaluation_flags_are_explicit():
    result = benchmark.make_result({"session_id": "s", "text": "same", "take_judge_verdict": "WINNER"},
        {"id": "n", "text": "same", "meta": {"take_judge_execution_status": "low_confidence"}}, "text", 1)
    assert result["take_judge_not_evaluated"] is True
    assert result["take_judge_low_confidence"] is True
    assert result["take_judge_unavailable"] is result["take_judge_abstained"] is False


def test_limit_metrics_use_only_selected_sessions():
    sessions = tuple(f"session-{index}" for index in range(5))
    s3 = S3([{"Key": f"Editdna good videos/{session}.mp4", "Size": 2000} for session in sessions], sessions=sessions)
    result = benchmark.run_benchmark("limited", {**request("inventory_only"), "limit": 3}, s3=s3,
                                     pipeline=lambda *_: pytest.fail("pipeline invoked"))
    summary = result["summary"]
    assert summary["dataset_total_historical_sessions"] == 5
    assert summary["dataset_total_historical_clips"] == 5
    assert summary["selected_historical_sessions"] == 3
    assert summary["selected_historical_clips"] == 3
    assert summary["total_historical_clips"] == 3
    assert summary["preflight"]["dataset_jsonl_readable"] is True


def test_checkpoint_stays_small_and_final_aggregation_loads_session_files():
    sessions = tuple(f"session-{index:03d}" for index in range(60))
    rows = [{"Key": f"Editdna good videos/{session}.mp4", "Size": 2000} for session in sessions]
    s3 = S3(rows, sessions=sessions); calls = []
    def pipeline(session, *_):
        calls.append(session)
        return {"clips": [{"id": session, "text": "hello product", "llm_reason": "x" * 50_000,
                           "slot": "HOOK", "meta": {"keep": True}}]}
    benchmark.run_benchmark("large", request(), s3=s3, pipeline=pipeline)
    checkpoint = s3.storage["editdna/benchmarks/large/checkpoint.json"]
    assert len(checkpoint) < 32_000
    assert b"x" * 100 not in checkpoint
    assert len(s3.storage["editdna/benchmarks/large/results_v2.jsonl"].splitlines()) == len(sessions)
    calls.clear()
    benchmark.run_benchmark("large", request(), s3=s3, pipeline=pipeline)
    assert calls == []


def test_session_output_keys_cannot_escape_benchmark_prefix():
    key = benchmark._session_key("editdna/benchmarks/job/", "../../secret session")
    assert key.startswith("editdna/benchmarks/job/sessions/") and ".." not in key
    benchmark_s3.validate_output_key(key, "job")
    with pytest.raises(ValueError):
        benchmark_s3.validate_output_key("editdna/benchmarks/job/sessions/../../secret.json", "job")


def test_production_defaults_and_request_scoped_activation(monkeypatch):
    from worker import pipeline
    signature = inspect.signature(pipeline.run_pipeline)
    assert signature.parameters["use_semantic_v2"].default is False
    assert signature.parameters["use_take_judge_v2"].default is False
    semantic_calls = []
    monkeypatch.setattr(pipeline, "EDITDNA_USE_LLM", False)
    monkeypatch.setattr(pipeline, "llm_classify_clips", lambda clips: semantic_calls.append(clips) or {})
    assert pipeline.enrich_clips_semantic([{"id": "x", "text": "hello", "meta": {}}]) is False
    assert semantic_calls == []
    pipeline.enrich_clips_semantic([{"id": "x", "text": "hello", "meta": {}}], force_v2=True)
    assert len(semantic_calls) == 1
    judge_calls = []
    monkeypatch.setattr(pipeline, "TAKE_JUDGE_ENABLED", False)
    monkeypatch.setattr(pipeline, "is_openai_available", lambda: True)
    monkeypatch.setattr(pipeline, "find_sibling_groups", lambda clips: judge_calls.append(clips) or [])
    pipeline.run_take_judge([], "/tmp", "input.mp4", force_enabled=True)
    assert judge_calls == [[]]


@pytest.mark.parametrize("fail", [False, True])
def test_benchmark_pipeline_does_not_retain_internal_directory(monkeypatch, tmp_path, fail):
    from worker import pipeline
    internal = tmp_path / ("failed-session" if fail else "successful-session")
    source = tmp_path / "source.mp4"; source.write_bytes(b"video")
    def session_dir(_): internal.mkdir(); return str(internal)
    monkeypatch.setattr(pipeline, "ensure_session_dir", session_dir)
    monkeypatch.setattr(pipeline, "probe_duration", lambda _: 1.0)
    if fail:
        monkeypatch.setattr(pipeline, "run_asr", lambda _: (_ for _ in ()).throw(RuntimeError("provider fallback failure")))
    else:
        monkeypatch.setattr(pipeline, "run_asr", lambda _: [{"id": "c", "start": 0, "end": 1, "text": "hello", "meta": {}}])
        monkeypatch.setattr(pipeline, "merge_incomplete_phrases", lambda clips: clips)
        monkeypatch.setattr(pipeline, "enrich_clips_semantic", lambda clips: False)
        monkeypatch.setattr(pipeline, "dedupe_clips", lambda clips: clips)
        monkeypatch.setattr(pipeline, "run_visual_pass", lambda *args: False)
        monkeypatch.setattr(pipeline, "build_slots_dict", lambda clips: {})
        monkeypatch.setattr(pipeline, "build_composer", lambda clips, mode: {"used_clip_ids": []})
        monkeypatch.setattr(pipeline, "pretty_print_composer", lambda *args: "")
    monkeypatch.setattr(pipeline, "S3_BUCKET", None)
    if fail:
        with pytest.raises(RuntimeError):
            pipeline.run_pipeline("benchmark", local_files=[str(source)], render_output=False,
                                  persist_result_json=False, retain_local_files=False)
    else:
        pipeline.run_pipeline("benchmark", local_files=[str(source)], render_output=False,
                              persist_result_json=False, retain_local_files=False)
    assert not internal.exists()


def test_authentication_required_and_missing_configuration_fails_closed(monkeypatch):
    client = TestClient(app)
    body = {**request("inventory_only"), "render_outputs": False}
    monkeypatch.delenv("BENCHMARK_INTERNAL_API_KEY", raising=False)
    assert client.post("/benchmark/run", json=body).status_code == 401
    monkeypatch.setenv("BENCHMARK_INTERNAL_API_KEY", "correct")
    assert client.post("/benchmark/run", json=body, headers={"X-Internal-API-Key": "wrong"}).status_code == 401
    fake = type("Job", (), {"id": "job"})()
    monkeypatch.setattr("web.routes_benchmark.enqueue_benchmark", lambda *args: (fake, False))
    assert client.post("/benchmark/run", json=body, headers={"X-Internal-API-Key": "correct"}).status_code == 202


def test_duplicate_job_prevention(monkeypatch):
    class Connection:
        def __init__(self): self.data = {}
        def get(self, key): return self.data.get(key)
        def set(self, key, value, **kwargs):
            if kwargs.get("nx") and key in self.data: return False
            self.data[key] = value; return True
        def delete(self, key): self.data.pop(key, None)
    class Queue:
        connection = Connection()
        def enqueue(self, *args, **kwargs): return type("Job", (), {"id": kwargs["job_id"]})()
    queue = Queue(); monkeypatch.setattr(jobs, "get_queue", lambda: queue)
    first, duplicate = jobs.enqueue_benchmark("one", request())
    assert first.id == "one" and duplicate is False
    active = type("Job", (), {"id": "one", "get_status": lambda self, refresh=False: type("S", (), {"value": "queued"})()})()
    monkeypatch.setattr(jobs.Job, "fetch", lambda *args, **kwargs: active)
    second, duplicate = jobs.enqueue_benchmark("two", request())
    assert second.id == "one" and duplicate is True


def test_rq_retries_cannot_be_lower_than_session_attempts(monkeypatch):
    monkeypatch.setenv("BENCHMARK_SESSION_MAX_ATTEMPTS", "4")
    monkeypatch.delenv("BENCHMARK_RQ_MAX_RETRIES", raising=False)
    assert jobs.benchmark_retry_count() == 3
    monkeypatch.setenv("BENCHMARK_RQ_MAX_RETRIES", "2")
    with pytest.raises(ValueError, match="cannot provide"):
        jobs.benchmark_retry_count()
    monkeypatch.setenv("BENCHMARK_RQ_MAX_RETRIES", "5")
    assert jobs.benchmark_retry_count() == 5


def test_inventory_preflight_write_failure_is_sanitized():
    class DeniedS3(S3):
        def put_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "raw provider detail"}}, "PutObject")
    s3 = DeniedS3([{"Key": "Editdna good videos/good-one.mp4", "Size": 2000}])
    result = benchmark.run_benchmark("denied", request("inventory_only"), s3=s3,
                                     pipeline=lambda *_: pytest.fail("pipeline invoked"))
    preflight = result["summary"]["preflight"]
    assert preflight["benchmark_write"] == {"successful": False, "error_classification": "access_denied"}
    assert preflight["benchmark_read_back"]["successful"] is False
    assert "raw provider detail" not in json.dumps(result)


def test_final_stage_remains_finished(monkeypatch):
    import tasks
    saved = []
    job = type("Job", (), {"meta": {}, "save_meta": lambda self: saved.append(dict(self.meta))})()
    monkeypatch.setattr("rq.get_current_job", lambda: job)
    monkeypatch.setattr(benchmark_s3, "client", lambda: S3())
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *args, **kwargs: {
        "summary": {"failed_sessions": 0}, "output_prefix": "editdna/benchmarks/job/"})
    tasks.job_benchmark("job", request("inventory_only"))
    assert saved[-1]["stage"] == "finished"
