"""RQ benchmark entry points executed inside the RunPod CutSell GPU worker."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from .config import load_runtime_config
from .universal_clean_cut_validation import run_single_universal_clean_cut_validation
from .validation import list_validation_videos

CLEAN_CUT_GOLD_PREFIX = "Editdna longform validation/"
BLOOPER_NEGATIVE_PREFIX = "Editdna bloopers videos/"
MAX_GOLD_VIDEOS = 32


def _safe_benchmark_id(value: object) -> str:
    raw = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", raw):
        raise ValueError("invalid benchmark_id")
    return raw


def _clean_cut_gold_prefix(value: object) -> str:
    prefix = str(value or CLEAN_CUT_GOLD_PREFIX).strip()
    if prefix == BLOOPER_NEGATIVE_PREFIX:
        raise ValueError(
            "bloopers-only data is negative-behavior training data and cannot be used as the Clean Cut gold selection benchmark"
        )
    if prefix != CLEAN_CUT_GOLD_PREFIX:
        raise ValueError(f"Clean Cut gold benchmark must use {CLEAN_CUT_GOLD_PREFIX!r}")
    return prefix


def _gold_video_limit(value: object) -> int:
    limit = 16 if value is None or value == "" else int(value)
    if not 1 <= limit <= MAX_GOLD_VIDEOS:
        raise ValueError(f"Clean Cut gold video_limit must be between 1 and {MAX_GOLD_VIDEOS}")
    return limit


def _upload_file(s3, *, bucket: str, key: str, path: str, content_type: str | None = None) -> str:
    extra = {"ContentType": content_type} if content_type else None
    if extra:
        s3.upload_file(path, bucket, key, ExtraArgs=extra)
    else:
        s3.upload_file(path, bucket, key)
    return f"s3://{bucket}/{key}"


def run_unseen_clean_cut_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the Clean Cut gold RAW suite on RunPod local perception plus approved Hybrid semantics.

    The gold suite is intentionally restricted to RAW long-form recordings that contain
    failed attempts/retries plus usable final delivery. Bloopers-only compilations are a
    separate negative-behavior dataset and are never valid winner-selection benchmarks.
    """
    from rq import get_current_job

    config = load_runtime_config()
    if config.brain_backend != "runpod_local":
        raise RuntimeError("unseen benchmark requires runpod_local brain")
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    benchmark_id = _safe_benchmark_id(payload.get("benchmark_id"))
    prefix = _clean_cut_gold_prefix(payload.get("source_prefix"))
    video_limit = _gold_video_limit(payload.get("video_limit"))
    expected_external = bool(payload.get("expected_external_brain_calls_enabled", False))

    import boto3
    s3 = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    keys = [item["key"] for item in list_validation_videos(prefix=prefix, limit=video_limit, s3=s3)]
    if not keys:
        raise RuntimeError(f"no Clean Cut gold RAW videos found under {prefix!r}")

    job = get_current_job()

    def publish(stage: str, percent: int, *, current_source: str | None = None) -> None:
        if job is None:
            return
        job.meta["stage"] = stage
        job.meta["progress_percent"] = max(0, min(100, int(percent)))
        job.meta["current_source"] = current_source
        job.meta["brain_backend"] = "runpod_local"
        job.meta["external_brain_calls_enabled"] = expected_external
        job.meta["dataset_role"] = "clean_cut_gold_raw"
        job.meta["source_prefix"] = prefix
        job.save_meta()

    artifact_prefix = f"cutsell/benchmarks/unseen/{benchmark_id}"
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    preview_uris: list[str] = []

    publish("starting", 1)
    with tempfile.TemporaryDirectory(prefix="cutsell-runpod-unseen-") as directory:
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
                    project_id=f"cutsell-unseen-{benchmark_id}-{index:03d}",
                    preview_output=str(preview_path),
                )
                if result.get("brain_backend") != "runpod_local":
                    raise RuntimeError("benchmark escaped runpod_local perception brain")
                if bool(result.get("external_brain_calls_enabled")) is not expected_external:
                    raise RuntimeError("benchmark Hybrid mode did not match requested contract")
                if expected_external:
                    if result.get("hybrid_provider") != "google":
                        raise RuntimeError("Hybrid benchmark did not use approved Google provider")
                    if result.get("hybrid_primary_model") != "gemini-3.5-flash-lite":
                        raise RuntimeError("Hybrid benchmark did not use approved primary model")
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
            "benchmark_suite": "clean_cut_gold_raw_generalization",
            "benchmark_id": benchmark_id,
            "dataset_role": "clean_cut_gold_raw",
            "source_prefix": prefix,
            "negative_behavior_prefix_excluded": BLOOPER_NEGATIVE_PREFIX,
            "brain_backend": "runpod_local",
            "external_brain_calls_enabled": expected_external,
            "hybrid_provider": "google" if expected_external else None,
            "hybrid_primary_model": "gemini-3.5-flash-lite" if expected_external else None,
            "source_keys": keys,
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

        report_path = Path(directory) / "cutsell-unseen-clean-cut-benchmark.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        report_uri = _upload_file(
            s3,
            bucket=config.s3_bucket,
            key=f"{artifact_prefix}/cutsell-unseen-clean-cut-benchmark.json",
            path=str(report_path),
            content_type="application/json",
        )

    publish("finished" if not failures and not provider_failures else "finished_with_failures", 100)
    return {
        "benchmark_id": benchmark_id,
        "dataset_role": report["dataset_role"],
        "source_prefix": report["source_prefix"],
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
