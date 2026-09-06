"""Focused Gold RAW validation for a small human-reviewed source set.

This job is intentionally separate from the 16-video generalization benchmark. It allows
CutSell to iterate on a small set of exact Gold RAW sources while preserving the same
RunPod-local perception and approved Google Hybrid semantics contract.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from .config import load_runtime_config
from .universal_clean_cut_validation import run_single_universal_clean_cut_validation
from .validation import list_validation_videos
from .validation_job import (
    CLEAN_CUT_GOLD_PREFIX,
    BLOOPER_NEGATIVE_PREFIX,
    _safe_benchmark_id,
    _clean_cut_gold_prefix,
    _upload_file,
)

MAX_FOCUSED_GOLD_VIDEOS = 6


def _requested_source_keys(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("focused Gold benchmark requires source_keys list")
    keys = tuple(str(item or "").strip() for item in value if str(item or "").strip())
    if not 1 <= len(keys) <= MAX_FOCUSED_GOLD_VIDEOS:
        raise ValueError(f"focused Gold benchmark requires 1-{MAX_FOCUSED_GOLD_VIDEOS} source_keys")
    if len(set(keys)) != len(keys):
        raise ValueError("focused Gold benchmark source_keys must be unique")
    if any(not key.startswith(CLEAN_CUT_GOLD_PREFIX) for key in keys):
        raise ValueError("focused Gold source key escaped Clean Cut Gold prefix")
    return keys


def run_focused_clean_cut_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Run exact requested Gold RAW sources and return the same artifact contract as Gold."""
    from rq import get_current_job

    config = load_runtime_config()
    if config.brain_backend != "runpod_local":
        raise RuntimeError("focused benchmark requires runpod_local brain")
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    benchmark_id = _safe_benchmark_id(payload.get("benchmark_id"))
    prefix = _clean_cut_gold_prefix(payload.get("source_prefix"))
    requested_keys = _requested_source_keys(payload.get("source_keys"))
    expected_external = bool(payload.get("expected_external_brain_calls_enabled", False))

    import boto3
    s3 = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    available = {
        item["key"] for item in list_validation_videos(prefix=prefix, limit=32, s3=s3)
    }
    missing = tuple(key for key in requested_keys if key not in available)
    if missing:
        raise RuntimeError(f"requested focused Gold sources not found: {missing!r}")
    keys = requested_keys

    job = get_current_job()

    def publish(stage: str, percent: int, *, current_source: str | None = None) -> None:
        if job is None:
            return
        job.meta["stage"] = stage
        job.meta["progress_percent"] = max(0, min(100, int(percent)))
        job.meta["current_source"] = current_source
        job.meta["brain_backend"] = "runpod_local"
        job.meta["external_brain_calls_enabled"] = expected_external
        job.meta["dataset_role"] = "clean_cut_gold_raw_focused"
        job.meta["source_prefix"] = prefix
        job.meta["source_keys"] = list(keys)
        job.save_meta()

    artifact_prefix = f"cutsell/benchmarks/focused/{benchmark_id}"
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    preview_uris: list[str] = []

    publish("starting", 1)
    with tempfile.TemporaryDirectory(prefix="cutsell-runpod-focused-") as directory:
        preview_dir = Path(directory) / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        source_count = len(keys)
        for index, key in enumerate(keys):
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", PurePosixPath(key).stem)[:80] or "video"
            preview_path = preview_dir / f"{index:02d}-{safe_name}.mp4"
            publish("processing", 3 + int(index * 84 / source_count), current_source=key)
            try:
                result = run_single_universal_clean_cut_validation(
                    key,
                    project_id=f"cutsell-focused-{benchmark_id}-{index:03d}",
                    preview_output=str(preview_path),
                )
                if result.get("brain_backend") != "runpod_local":
                    raise RuntimeError("focused benchmark escaped runpod_local perception brain")
                if bool(result.get("external_brain_calls_enabled")) is not expected_external:
                    raise RuntimeError("focused benchmark Hybrid mode did not match requested contract")
                if expected_external:
                    if result.get("hybrid_provider") != "google":
                        raise RuntimeError("focused benchmark did not use approved Google provider")
                    if result.get("hybrid_primary_model") != "gemini-3.5-flash-lite":
                        raise RuntimeError("focused benchmark used unexpected primary model")
                results.append(result)
                if preview_path.exists():
                    preview_key = f"{artifact_prefix}/previews/{preview_path.name}"
                    preview_uris.append(_upload_file(
                        s3,
                        bucket=config.s3_bucket,
                        key=preview_key,
                        path=str(preview_path),
                        content_type="video/mp4",
                    ))
            except Exception as exc:
                failures.append({
                    "source_key": key,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:500],
                })

        total_input = sum(float(item.get("source_duration_sec") or 0) for item in results)
        total_selected = sum(float(item.get("selected_duration_sec") or 0) for item in results)
        total_hybrid_requested = sum(int(item.get("hybrid_requested_group_count") or 0) for item in results)
        total_hybrid_available = sum(int(item.get("hybrid_available_group_count") or 0) for item in results)
        total_hybrid_deleted = sum(int(item.get("hybrid_deleted_count") or 0) for item in results)
        provider_failures = []
        if expected_external and total_hybrid_requested and total_hybrid_available == 0:
            provider_failures.append({"provider": "google", "error": "Hybrid judge requested but no successful groups"})

        report = {
            "benchmark_suite": "clean_cut_gold_raw_focused",
            "benchmark_id": benchmark_id,
            "dataset_role": "clean_cut_gold_raw_focused",
            "source_prefix": prefix,
            "negative_behavior_prefix_excluded": BLOOPER_NEGATIVE_PREFIX,
            "brain_backend": "runpod_local",
            "external_brain_calls_enabled": expected_external,
            "hybrid_provider": "google" if expected_external else None,
            "hybrid_primary_model": "gemini-3.5-flash-lite" if expected_external else None,
            "source_keys": list(keys),
            "source_count": len(keys),
            "completed_count": len(results),
            "execution_failure_count": len(failures),
            "provider_failure_count": len(provider_failures),
            "total_hybrid_requested_groups": total_hybrid_requested,
            "total_hybrid_available_groups": total_hybrid_available,
            "total_hybrid_deleted": total_hybrid_deleted,
            "total_input_duration_sec": round(total_input, 3),
            "total_selected_duration_sec": round(total_selected, 3),
            "selected_to_input_ratio": round(total_selected / total_input, 4) if total_input else None,
            "total_selected": sum(int(item.get("selected_count") or 0) for item in results),
            "total_alternates": sum(int(item.get("alternate_count") or 0) for item in results),
            "total_discarded": sum(int(item.get("discarded_count") or 0) for item in results),
            "total_clean_cut_removed": sum(int(item.get("clean_cut_removed_count") or 0) for item in results),
            "total_temporal_trimmed": sum(int(item.get("temporal_trimmed_count") or 0) for item in results),
            "preview_uris": preview_uris,
            "results": results,
            "failures": failures,
            "provider_failures": provider_failures,
        }

        report_path = Path(directory) / "cutsell-focused-clean-cut-benchmark.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        report_uri = _upload_file(
            s3,
            bucket=config.s3_bucket,
            key=f"{artifact_prefix}/cutsell-focused-clean-cut-benchmark.json",
            path=str(report_path),
            content_type="application/json",
        )

    publish("finished" if not failures and not provider_failures else "finished_with_failures", 100)
    return {
        "benchmark_id": benchmark_id,
        "dataset_role": report["dataset_role"],
        "source_prefix": report["source_prefix"],
        "source_keys": list(keys),
        "brain_backend": "runpod_local",
        "external_brain_calls_enabled": expected_external,
        "hybrid_provider": report["hybrid_provider"],
        "hybrid_primary_model": report["hybrid_primary_model"],
        "report_uri": report_uri,
        "preview_uris": preview_uris,
        "source_count": len(keys),
        "completed_count": len(results),
        "execution_failure_count": len(failures),
        "provider_failure_count": len(provider_failures),
        "total_hybrid_requested_groups": total_hybrid_requested,
        "total_hybrid_available_groups": total_hybrid_available,
        "total_hybrid_deleted": total_hybrid_deleted,
        "selected_to_input_ratio": report["selected_to_input_ratio"],
        "total_clean_cut_removed": report["total_clean_cut_removed"],
        "total_temporal_trimmed": report["total_temporal_trimmed"],
    }
