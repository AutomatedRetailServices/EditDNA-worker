"""RunPod Serverless entrypoint for the CutSell GPU brain.

This is intentionally thin: it reuses the exact CutSell validation/brain code and only
adapts the execution surface from a long-lived RQ worker to queue-based Serverless.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
import runpod

from .universal_clean_cut_validation import run_single_universal_clean_cut_validation


def _upload_artifact(local_path: str, *, key: str, content_type: str) -> str:
    bucket = str(os.environ.get("S3_BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET is required")
    region = str(os.environ.get("AWS_REGION") or "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{key}"


def _health() -> dict:
    # Keep CUDA/PyTorch as a runtime-only dependency so the clean-worker/API import
    # boundary stays free of heavy ML imports. The Serverless image provides torch.
    import torch

    return {
        "ok": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _focused(payload: dict) -> dict:
    source_key = str(payload.get("source_key") or "").strip()
    if not source_key:
        raise ValueError("source_key is required")
    benchmark_id = str(payload.get("benchmark_id") or "serverless-focused").strip()
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in benchmark_id)[:100]
    work = Path("/tmp/cutsell-serverless")
    work.mkdir(parents=True, exist_ok=True)
    preview = work / f"{safe_id}.mp4"
    result_path = work / f"{safe_id}.json"
    result = run_single_universal_clean_cut_validation(
        source_key,
        project_id=safe_id,
        preview_output=str(preview),
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    prefix = f"cutsell/serverless/{safe_id}"
    preview_uri = _upload_artifact(str(preview), key=f"{prefix}/preview.mp4", content_type="video/mp4")
    result_uri = _upload_artifact(str(result_path), key=f"{prefix}/result.json", content_type="application/json")
    return {
        "ok": True,
        "benchmark_id": safe_id,
        "source_key": source_key,
        "brain_backend": result.get("brain_backend"),
        "external_brain_calls_enabled": result.get("external_brain_calls_enabled"),
        "selected_count": result.get("selected_count"),
        "discarded_count": result.get("discarded_count"),
        "elapsed_sec": result.get("elapsed_sec"),
        "preview_uri": preview_uri,
        "result_uri": result_uri,
    }


def handler(job: dict) -> dict:
    payload = dict(job.get("input") or {})
    op = str(payload.get("op") or "health").strip().lower()
    if op == "health":
        return _health()
    if op == "focused":
        return _focused(payload)
    raise ValueError(f"unsupported op: {op}")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
