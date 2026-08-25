"""RunPod Serverless entrypoint for the CutSell GPU brain."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import boto3
import runpod

from .locked_selection_replay import run_locked_selection_replay, _apply_review_cuts, _probe_duration
from .speech_visual_microtrim import detect_speech_safe_visual_microtrims
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
    torch = importlib.import_module("torch")
    return {
        "ok": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _safe_id(value: str, fallback: str) -> str:
    raw = str(value or fallback).strip()
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)[:100]


def _focused(payload: dict) -> dict:
    source_key = str(payload.get("source_key") or "").strip()
    if not source_key:
        raise ValueError("source_key is required")
    safe_id = _safe_id(payload.get("benchmark_id"), "serverless-focused")
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


def _locked_selection(payload: dict) -> dict:
    source_key = str(payload.get("source_key") or "").strip()
    if not source_key:
        raise ValueError("source_key is required")
    selection = payload.get("selection")
    if not isinstance(selection, list) or not selection:
        raise ValueError("selection must be a non-empty list")
    review_cuts = payload.get("review_cuts")
    if review_cuts is not None and not isinstance(review_cuts, list):
        raise ValueError("review_cuts must be a list when provided")
    auto_microtrim = bool(payload.get("auto_speech_visual_microtrim", False))

    safe_id = _safe_id(payload.get("benchmark_id"), "serverless-locked-selection")
    work = Path("/tmp/cutsell-serverless")
    work.mkdir(parents=True, exist_ok=True)
    preview = work / f"{safe_id}.mp4"
    result_path = work / f"{safe_id}.json"
    result = run_locked_selection_replay(
        source_key,
        selection,
        project_id=safe_id,
        preview_output=str(preview),
        review_cuts=review_cuts,
    )

    auto_cuts = ()
    auto_diag = {
        "speech_lock_ok": True,
        "auto_microtrim_count": 0,
        "auto_microtrim_duration_sec": 0.0,
        "frame_aware": True,
        "rule": "disabled",
    }
    if auto_microtrim:
        auto_cuts, auto_diag = detect_speech_safe_visual_microtrims(
            str(preview),
            asr_model=str(os.environ.get("CUTSELL_ASR_MODEL") or "medium"),
        )
        if auto_cuts:
            _apply_review_cuts(str(preview), auto_cuts)

    result = {
        **result,
        "output_duration_sec": round(_probe_duration(str(preview)), 3),
        "auto_speech_visual_microtrim_enabled": auto_microtrim,
        "auto_microtrim_count": len(auto_cuts),
        "auto_microtrim_duration_sec": round(sum(float(c["end"]) - float(c["start"]) for c in auto_cuts), 3),
        "auto_microtrims": list(auto_cuts),
        "auto_microtrim_diagnostics": auto_diag,
        "speech_lock_ok": bool(auto_diag.get("speech_lock_ok", True)),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    prefix = f"cutsell/serverless/{safe_id}"
    preview_uri = _upload_artifact(str(preview), key=f"{prefix}/preview.mp4", content_type="video/mp4")
    result_uri = _upload_artifact(str(result_path), key=f"{prefix}/result.json", content_type="application/json")
    return {
        "ok": True,
        "benchmark_id": safe_id,
        "source_key": source_key,
        "selection_authority": result.get("selection_authority"),
        "review_cut_authority": result.get("review_cut_authority"),
        "external_brain_calls_enabled": False,
        "selected_count": result.get("selected_count"),
        "selected_duration_sec": result.get("selected_duration_sec"),
        "baseline_output_duration_sec": result.get("baseline_output_duration_sec"),
        "output_duration_sec": result.get("output_duration_sec"),
        "review_cut_count": result.get("review_cut_count"),
        "review_cut_duration_sec": result.get("review_cut_duration_sec"),
        "auto_microtrim_count": result.get("auto_microtrim_count"),
        "auto_microtrim_duration_sec": result.get("auto_microtrim_duration_sec"),
        "speech_lock_ok": result.get("speech_lock_ok"),
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
    if op == "locked_selection":
        return _locked_selection(payload)
    raise ValueError(f"unsupported op: {op}")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
