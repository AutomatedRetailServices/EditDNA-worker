import io
import json

import pytest

import benchmark
import benchmark_s3


class S3:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.puts = []
        self.presigns = []

    def list_objects_v2(self, **kwargs): return {"Contents": self.rows, "IsTruncated": False}
    def get_object(self, **kwargs):
        data = (json.dumps({"session_id": "good-one", "clip_id": "old", "text": "hello product", "keep": True, "slot": "HOOK"}) + "\n").encode()
        return {"ContentLength": len(data), "Body": io.BytesIO(data)}
    def put_object(self, **kwargs): self.puts.append(kwargs)
    def generate_presigned_url(self, *args, **kwargs): self.presigns.append(kwargs); return "signed"


@pytest.fixture(autouse=True)
def bucket(monkeypatch): monkeypatch.setenv("S3_BUCKET", "private-test-bucket")


def test_allowlists_block_arbitrary_bucket_keys_and_traversal():
    assert benchmark_s3.validate_input_prefix("Editdna good videos/")
    with pytest.raises(ValueError): benchmark_s3.validate_input_prefix("private/")
    with pytest.raises(ValueError): benchmark_s3.validate_dataset_key("editdna/training/../secret.jsonl")
    with pytest.raises(ValueError): benchmark_s3.validate_dataset_key("other/data.jsonl")


def test_listing_filters_folders_dot_underscore_tiny_and_extensions(monkeypatch):
    monkeypatch.setattr(benchmark_s3, "MIN_OBJECT_BYTES", 100)
    s3 = S3([{"Key": "Editdna good videos/", "Size": 0}, {"Key": "Editdna good videos/._x.mp4", "Size": 500},
             {"Key": "Editdna good videos/tiny.mp4", "Size": 2}, {"Key": "Editdna good videos/x.txt", "Size": 500},
             {"Key": "Editdna good videos/ok.mp4", "Size": 500}])
    assert benchmark_s3.list_objects(s3, "Editdna good videos/") == [{"key": "Editdna good videos/ok.mp4", "size": 500}]


def test_deterministic_and_ambiguous_source_resolution():
    grouped = {"good-one": [{}], "same": [{}]}
    objects = [{"key": "Editdna good videos/good-one.mp4"}, {"key": "Editdna good videos/same-a.mp4"}, {"key": "Editdna good videos/same-b.mp4"}]
    resolved, unresolved = benchmark.resolve_sources(grouped, objects)
    assert resolved["good-one"]["key"].endswith("good-one.mp4")
    assert unresolved["same"]["classification"] == "ambiguous_source"


def test_failure_continues_resume_and_no_historical_overwrite_or_presign():
    s3 = S3([{"Key": "Editdna good videos/good-one.mp4", "Size": 2000}])
    calls = []
    def failed(*args): calls.append(args); raise RuntimeError("safe failure")
    result = benchmark.run_benchmark("job-1", {"dataset_key": "editdna/training/take_judge_dataset.jsonl",
        "source_prefixes": ["Editdna good videos/"], "mode": "old_vs_new", "completed_sessions": []}, s3=s3, pipeline=failed)
    assert result["summary"]["failed_sessions"] == 1
    assert all(item["Key"].startswith("editdna/benchmarks/job-1/") for item in s3.puts)
    assert not s3.presigns
    calls.clear()
    benchmark.run_benchmark("job-2", {"dataset_key": "editdna/training/take_judge_dataset.jsonl",
        "source_prefixes": ["Editdna good videos/"], "mode": "old_vs_new", "completed_sessions": ["good-one"]}, s3=s3, pipeline=failed)
    assert calls == []


def test_inventory_only_never_invokes_pipeline():
    s3 = S3([{"Key": "Editdna good videos/good-one.mp4", "Size": 2000}])
    benchmark.run_benchmark("dry", {"dataset_key": "editdna/training/take_judge_dataset.jsonl",
        "source_prefixes": ["Editdna good videos/"], "mode": "inventory_only"}, s3=s3,
        pipeline=lambda *args: pytest.fail("pipeline invoked"))
