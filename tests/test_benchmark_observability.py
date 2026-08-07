import json

from worker.benchmark_observability import (
    collect_semantic_execution_summary,
    embed_semantic_execution,
)


def test_embed_semantic_execution_preserves_existing_provider_usage_fields():
    result = {
        "provider_usage_instrumentation": {"available": False, "reason": "not instrumented"},
        "semantic_execution": {
            "requested": True,
            "provider_available": True,
            "status_counts": {"applied": 1},
            "clip_status_counts": {"applied": 2},
            "fallback_reasons": {},
        },
    }
    embedded = embed_semantic_execution(result)
    usage = embedded["provider_usage_instrumentation"]
    assert usage["available"] is False
    assert usage["reason"] == "not instrumented"
    assert usage["semantic_execution"]["status_counts"] == {"applied": 1}


def test_collect_semantic_execution_summary_aggregates_completed_sessions():
    prefix = "editdna/benchmarks/job/"
    storage = {
        prefix + "checkpoint.json": json.dumps({
            "session_result_keys": {
                "one": prefix + "sessions/one.json",
                "two": prefix + "sessions/two.json",
            }
        }).encode(),
        prefix + "sessions/one.json": json.dumps({
            "provider_instrumentation": {"usage": {"semantic_execution": {
                "requested": True,
                "provider_available": True,
                "status_counts": {"applied": 1},
                "clip_status_counts": {"applied": 3, "abstained": 1},
                "fallback_reasons": {"model_abstained": 1},
            }}}
        }).encode(),
        prefix + "sessions/two.json": json.dumps({
            "provider_instrumentation": {"usage": {"semantic_execution": {
                "requested": True,
                "provider_available": False,
                "status_counts": {"provider_unavailable": 1},
                "clip_status_counts": {"provider_unavailable": 2},
                "fallback_reasons": {"provider_unavailable": 2},
            }}}
        }).encode(),
    }

    def read_output(_s3, key, _job_id, max_bytes=None):
        return storage.get(key)

    summary = collect_semantic_execution_summary(object(), "job", prefix, read_output)
    observed = summary["semantic_execution_observability"]
    assert observed["available"] is True
    assert observed["observed_sessions"] == 2
    assert observed["missing_sessions"] == 0
    assert observed["requested_sessions"] == 2
    assert observed["provider_available_sessions"] == 1
    assert observed["source_status_counts"] == {"applied": 1, "provider_unavailable": 1}
    assert observed["clip_status_counts"] == {
        "applied": 3,
        "abstained": 1,
        "provider_unavailable": 2,
    }
    assert observed["fallback_reasons"] == {
        "model_abstained": 1,
        "provider_unavailable": 2,
    }


def test_collect_semantic_execution_summary_fails_open_when_checkpoint_missing():
    summary = collect_semantic_execution_summary(
        object(), "job", "editdna/benchmarks/job/", lambda *_args, **_kwargs: None
    )
    assert summary == {
        "semantic_execution_observability": {
            "available": False,
            "reason": "benchmark_session_metadata_unavailable",
        }
    }
