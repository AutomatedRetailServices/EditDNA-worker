"""Benchmark aggregation helpers for optional intelligence observability.

The benchmark orchestrator predates Semantic V2 execution metadata. These
helpers preserve that metadata inside the already persisted provider-usage
payload and aggregate it after the resumable benchmark finishes, without
changing clip decisions or benchmark matching behavior.
"""
from collections import Counter
import json
from typing import Any, Callable, Dict


ReadOutput = Callable[..., bytes | None]


def embed_semantic_execution(result: Dict[str, Any]) -> Dict[str, Any]:
    """Persist Semantic execution metadata through the legacy benchmark payload."""
    execution = result.get("semantic_execution")
    if not isinstance(execution, dict):
        return result

    usage = result.get("provider_usage_instrumentation")
    if not isinstance(usage, dict):
        usage = {
            "available": False,
            "reason": "Provider token usage is not instrumented",
        }
        result["provider_usage_instrumentation"] = usage
    usage["semantic_execution"] = execution
    return result


def collect_semantic_execution_summary(
    s3: Any,
    job_id: str,
    output_prefix: str,
    read_output: ReadOutput,
) -> Dict[str, Any]:
    """Aggregate Semantic execution from completed per-session benchmark files."""
    checkpoint_key = output_prefix + "checkpoint.json"
    try:
        raw_checkpoint = read_output(s3, checkpoint_key, job_id, max_bytes=4 * 1024 * 1024)
        if raw_checkpoint is None:
            raise ValueError("checkpoint missing")
        checkpoint = json.loads(raw_checkpoint)
    except Exception:
        return {
            "semantic_execution_observability": {
                "available": False,
                "reason": "benchmark_session_metadata_unavailable",
            }
        }

    session_keys = checkpoint.get("session_result_keys") or {}
    source_statuses = Counter()
    clip_statuses = Counter()
    fallback_reasons = Counter()
    observed_sessions = 0
    missing_sessions = 0
    provider_available_sessions = 0
    requested_sessions = 0

    for session_id in sorted(session_keys):
        key = session_keys[session_id]
        try:
            raw = read_output(s3, key, job_id, max_bytes=64 * 1024 * 1024)
            item = json.loads(raw) if raw else {}
        except Exception:
            missing_sessions += 1
            continue

        usage = (item.get("provider_instrumentation") or {}).get("usage") or {}
        execution = usage.get("semantic_execution")
        if not isinstance(execution, dict):
            missing_sessions += 1
            continue

        observed_sessions += 1
        if execution.get("requested"):
            requested_sessions += 1
        if execution.get("provider_available"):
            provider_available_sessions += 1
        source_statuses.update(execution.get("status_counts") or {})
        clip_statuses.update(execution.get("clip_status_counts") or {})
        fallback_reasons.update(execution.get("fallback_reasons") or {})

    return {
        "semantic_execution_observability": {
            "available": observed_sessions > 0,
            "observed_sessions": observed_sessions,
            "missing_sessions": missing_sessions,
            "requested_sessions": requested_sessions,
            "provider_available_sessions": provider_available_sessions,
            "source_status_counts": dict(source_statuses),
            "clip_status_counts": dict(clip_statuses),
            "fallback_reasons": dict(fallback_reasons),
        }
    }
