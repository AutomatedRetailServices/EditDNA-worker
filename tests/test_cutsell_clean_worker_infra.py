from types import SimpleNamespace

from cutsell_worker.queueing import enqueue_flow_b
from cutsell_worker.serde import request_from_dict
from cutsell_worker.storage import parse_s3_uri


def test_parse_s3_uri_requires_bucket_and_key():
    assert parse_s3_uri("s3://bucket/path/raw.mov") == ("bucket", "path/raw.mov")
    for bad in ("https://example.com/a.mov", "s3://bucket", "s3:///key"):
        try:
            parse_s3_uri(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid S3 URI must be rejected")


def test_request_from_dict_supports_multiple_sources_and_order():
    request = request_from_dict({
        "project_id": "p1",
        "user_id": "u1",
        "language_hint": "es",
        "sources": [
            {"source_asset_id": "s1", "uri": "s3://b/1.mov", "original_name": "1.mov", "source_order": 1},
            {"source_asset_id": "s0", "uri": "s3://b/0.mov", "original_name": "0.mov", "source_order": 0},
        ],
    })
    assert len(request.sources) == 2
    assert request.language_hint == "es"
    assert [source.source_asset_id for source in request.sources] == ["s1", "s0"]


class FakeQueue:
    name = "cutsell"
    def __init__(self):
        self.calls = []
    def enqueue(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(id="job-123")


def test_enqueue_flow_b_targets_clean_worker_job():
    queue = FakeQueue()
    submitted = enqueue_flow_b({"project_id": "p1"}, queue=queue)
    assert submitted.job_id == "job-123"
    assert submitted.queue_name == "cutsell"
    assert queue.calls[0][0][0] == "cutsell_worker.worker_job.run_flow_b_job"
