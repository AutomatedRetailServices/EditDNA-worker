import inspect
import io
import json

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
    calls = []
    benchmark.run_benchmark("job", request(), s3=s3, pipeline=lambda *args: calls.append(args))
    assert calls == []  # checkpoint automatically prevents reruns


def test_failure_continues_and_summary_metrics_are_explicit():
    rows = [{"Key": "Editdna good videos/good-one.mp4", "Size": 2000},
            {"Key": "Editdna good videos/good-two.mp4", "Size": 2000}]
    s3 = S3(rows, sessions=("good-one", "good-two")); calls = []
    def pipeline(session, *_):
        calls.append(session)
        if session == "good-one": raise RuntimeError("safe failure")
        return {"clips": [{"id": "new", "text": "hello product", "slot": "BENEFITS", "meta": {"keep": False}}],
                "provider_usage": {"openai_calls": 2}, "estimated_cost": .25}
    result = benchmark.run_benchmark("job", request(), s3=s3, pipeline=pipeline)
    assert calls == ["good-one", "good-two"]
    summary = result["summary"]
    assert summary["failed_sessions"] == 1
    assert summary["per_slot_changes"] == {"HOOK->BENEFITS": 1}
    assert summary["per_source_session_changes"] == {"good-two": 1}
    assert summary["provider_usage"] == {"openai_calls": 2}
    assert summary["estimated_cost"] == .25 and summary["estimated_cost_metadata"]["available"] is True


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
