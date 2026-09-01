"""RunPod Serverless entrypoint for the CutSell GPU brain."""
from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
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


def _safe_hostname() -> str | None:
    """Best-effort worker/container identifier. Never raises."""
    try:
        return socket.gethostname()
    except Exception:
        return None


def _safe_worker_id() -> str | None:
    """RunPod exposes the pod id via this env var on serverless workers.
    Best-effort only -- absent outside a real RunPod worker."""
    try:
        return os.environ.get("RUNPOD_POD_ID") or None
    except Exception:
        return None


def _safe_nvidia_driver_cuda_version() -> str | None:
    """CUDA *runtime* (driver-reported) version, distinct from the CUDA
    version torch was *compiled* against (`torch.version.cuda`) -- the two
    can differ, and a driver/runtime mismatch is exactly the kind of
    evidence a bare `cuda_available=false` collapses away. Best-effort via
    `nvidia-smi`; short timeout, fails safely to None on any error (binary
    missing, no GPU, permission denied, timeout, unexpected output)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        return line or None
    except Exception:
        return None


def _diagnose_capability_incompatibility(torch_module, compute_capability: str) -> str:
    """Explicit diagnosis for "this torch build predates the assigned GPU's
    compute capability" (D-041 GPU-fallback-audit follow-up -- the RTX PRO
    6000 Blackwell / sm_120 case), so a future health result says this
    plainly instead of collapsing to a bare cuda_available=false. Compares
    against torch's own reported supported-architecture list rather than
    scraping warning text, which is fragile across torch versions. Fails
    safely to a still-informative message if the arch list itself can't be
    read."""
    try:
        supported = tuple(torch_module.cuda.get_arch_list())
    except Exception:
        return (
            f"cuda_available is False and a device with compute capability "
            f"{compute_capability} was detected, but this torch build's supported "
            f"architecture list could not be read to confirm why."
        )
    if compute_capability not in supported:
        supported_desc = ", ".join(supported) if supported else "none"
        return (
            f"Detected GPU compute capability {compute_capability} is not in this "
            f"torch build's supported architecture list ({supported_desc}). The "
            f"current torch/CUDA build does not support this GPU."
        )
    return (
        f"cuda_available is False even though compute capability {compute_capability} "
        f"appears in this torch build's supported architecture list "
        f"({', '.join(supported)}); the incompatibility is not explained by "
        f"architecture support alone."
    )


def _health() -> dict:
    """Diagnostic-only (D-041 GPU-fallback-audit follow-up). Every hardware/
    runtime read is independently guarded so one failing probe (e.g. an
    incompatible GPU's device-name lookup) never hides the others -- the
    goal is that an incompatible GPU such as RTX PRO 6000 Blackwell (sm_120)
    reports explicitly why, instead of collapsing to a bare
    cuda_available=false. `ok`/`cuda_available` keep their existing meaning
    (runpod_orchestration.py's health classification reads both unchanged)."""
    torch = importlib.import_module("torch")
    result: dict = {
        "ok": True,
        "torch_version": torch.__version__,
        "torch_compiled_cuda_version": torch.version.cuda,
        "hostname": _safe_hostname(),
        "worker_id": _safe_worker_id(),
    }

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        result["cuda_init_error"] = f"{type(exc).__name__}: {exc}"
    result["cuda_available"] = cuda_available

    device_count = 0
    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        result["cuda_device_count_error"] = f"{type(exc).__name__}: {exc}"
    result["cuda_device_count"] = device_count

    device_name = None
    if device_count > 0:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            result["device_name_error"] = f"{type(exc).__name__}: {exc}"
    result["device_name"] = device_name

    compute_capability = None
    if device_count > 0:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"sm_{major}{minor}"
        except Exception as exc:
            result["compute_capability_error"] = f"{type(exc).__name__}: {exc}"
    result["compute_capability"] = compute_capability

    result["cuda_runtime_version"] = _safe_nvidia_driver_cuda_version()

    result["incompatibility_reason"] = (
        _diagnose_capability_incompatibility(torch, compute_capability)
        if (not cuda_available and compute_capability is not None)
        else None
    )

    return result


def _safe_id(value: str, fallback: str) -> str:
    raw = str(value or fallback).strip()
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)[:100]


def _focused(payload: dict) -> dict:
    source_key = str(payload.get("source_key") or "").strip()
    if not source_key:
        raise ValueError("source_key is required")
    auto_microtrim = bool(payload.get("auto_speech_visual_microtrim", False))
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

    # D-036 item 7: the ONE authoritative delivery gate -- a candidate this
    # harness rendered is deliverable if and only if the live render/QC
    # service (shared with the real export job) reached PASS. Everything
    # below (auto-microtrim, which S3 key the file is uploaded under, which
    # compact fields are populated) branches off this single value.
    live_qc = result.get("live_render_qc") or {}
    deliverable = bool(live_qc.get("deliverable"))
    delivery_status = str(live_qc.get("delivery_status") or "NOT_DELIVERABLE_unknown")

    auto_cuts = ()
    auto_diag = {"speech_lock_ok": True, "auto_microtrim_count": 0, "auto_microtrim_duration_sec": 0.0, "frame_aware": True, "rule": "disabled"}
    if auto_microtrim and deliverable and preview.exists():
        # Never spend ASR/microtrim work refining a candidate that will not
        # be delivered anyway -- an invalidated render stays exactly as QC
        # left it, for diagnosis.
        auto_cuts, auto_diag = detect_speech_safe_visual_microtrims(
            str(preview),
            asr_model=str(os.environ.get("CUTSELL_ASR_MODEL") or "medium"),
        )
        if auto_cuts:
            _apply_review_cuts(str(preview), auto_cuts)

    result = {
        **result,
        "output_duration_sec": round(_probe_duration(str(preview)), 3) if preview.exists() else None,
        "auto_speech_visual_microtrim_enabled": auto_microtrim,
        "auto_microtrim_count": len(auto_cuts),
        "auto_microtrim_duration_sec": round(sum(float(c["end"]) - float(c["start"]) for c in auto_cuts), 3),
        "auto_microtrims": list(auto_cuts),
        "auto_microtrim_diagnostics": auto_diag,
        "speech_lock_ok": bool(auto_diag.get("speech_lock_ok", True)),
        "deliverable": deliverable,
        "delivery_status": delivery_status,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    prefix = f"cutsell/serverless/{safe_id}"

    # D-036 item 6: a candidate invalidated by QC must never be surfaced as
    # if it were deliverable. A PASSing candidate uploads under the normal
    # `preview.mp4` name (unchanged behavior); anything else is preserved
    # ONLY as a clearly-named diagnostic artifact, with `preview_uri` left
    # null so nothing downstream mistakes it for the real thing.
    preview_uri = None
    diagnostic_preview_uri = None
    if preview.exists():
        if deliverable:
            preview_uri = _upload_artifact(str(preview), key=f"{prefix}/preview.mp4", content_type="video/mp4")
        else:
            diagnostic_preview_uri = _upload_artifact(
                str(preview), key=f"{prefix}/diagnostic-invalidated-preview.mp4", content_type="video/mp4",
            )
    result_uri = _upload_artifact(str(result_path), key=f"{prefix}/result.json", content_type="application/json")
    return {
        "ok": True,
        "benchmark_id": safe_id,
        "source_key": source_key,
        "brain_backend": result.get("brain_backend"),
        "external_brain_calls_enabled": result.get("external_brain_calls_enabled"),
        "selection_reasoner_enabled": result.get("selection_reasoner_enabled"),
        "selection_reasoner_status": result.get("selection_reasoner_status"),
        "selection_reasoner_provider": result.get("selection_reasoner_provider"),
        "selection_reasoner_model": result.get("selection_reasoner_model"),
        "hybrid_requested_group_count": result.get("hybrid_requested_group_count"),
        "selected_count": result.get("selected_count"),
        "alternate_count": result.get("alternate_count"),
        "discarded_count": result.get("discarded_count"),
        "selected_duration_sec": result.get("selected_duration_sec"),
        "elapsed_sec": result.get("elapsed_sec"),
        "output_duration_sec": result.get("output_duration_sec"),
        "auto_microtrim_count": result.get("auto_microtrim_count"),
        "auto_microtrim_duration_sec": result.get("auto_microtrim_duration_sec"),
        "speech_lock_ok": result.get("speech_lock_ok"),
        # D-030/D-035/D-036: this preview was rendered through the exact same
        # live PostRenderWatchListenQC + bounded physical repair service the
        # real export job uses -- surface its outcome, and the authoritative
        # delivery gate, even in the compact RunPod summary so a run's
        # deliverability is visible without downloading the full result.json.
        "preview_skipped_reason": result.get("preview_skipped_reason"),
        "deliverable": deliverable,
        "delivery_status": delivery_status,
        "live_render_qc_status": live_qc.get("status"),
        "live_render_qc_render_attempt_count": live_qc.get("render_attempt_count"),
        "live_render_qc_plan_id": live_qc.get("plan_id"),
        "live_render_qc_plan_version": live_qc.get("plan_version"),
        "live_render_qc_semantic_hash": live_qc.get("semantic_hash"),
        # NOT DELIVERABLE / QC INVALIDATED when set -- diagnostic only, never
        # the final candidate.
        "preview_uri": preview_uri,
        "diagnostic_preview_uri": diagnostic_preview_uri,
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
