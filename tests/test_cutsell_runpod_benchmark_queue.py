from __future__ import annotations

from cutsell_worker.queueing import enqueue_unseen_clean_cut_benchmark
from cutsell_worker.validation_job import _safe_benchmark_id


class _Job:
    id = "job-runpod-benchmark"


class _Queue:
    name = "cutsell"

    def __init__(self):
        self.calls = []

    def enqueue(self, target, payload, **kwargs):
        self.calls.append((target, payload, kwargs))
        return _Job()


def test_unseen_benchmark_is_queued_to_runpod_validation_job():
    queue = _Queue()
    submission = enqueue_unseen_clean_cut_benchmark(
        {"benchmark_id": "unseen-17", "video_limit": 8},
        queue=queue,
    )

    assert submission.job_id == "job-runpod-benchmark"
    assert submission.queue_name == "cutsell"
    assert len(queue.calls) == 1
    target, payload, kwargs = queue.calls[0]
    assert target == "cutsell_worker.validation_job.run_unseen_clean_cut_benchmark"
    assert payload["benchmark_id"] == "unseen-17"
    assert kwargs["meta"]["brain_backend"] == "runpod_local"
    assert kwargs["meta"]["external_brain_calls_enabled"] is False
    assert kwargs["job_timeout"] == 10800


def test_benchmark_id_rejects_path_or_shell_like_values():
    assert _safe_benchmark_id("unseen-17.sha") == "unseen-17.sha"
    for bad in ("../unseen", "unseen/17", "unseen 17", "$(whoami)", ""):
        try:
            _safe_benchmark_id(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid benchmark id: {bad!r}")
