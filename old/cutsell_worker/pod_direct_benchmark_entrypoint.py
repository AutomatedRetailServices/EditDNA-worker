"""RunPod Pod direct-execution container entrypoint (D-042 follow-up:
"restore the known-working execution model" -- bypass the HTTP
pod_job_server/port-8080 transport entirely for this QA path).

This is the Pod's `dockerStartCmd` for the direct-execution benchmark
path: it runs ONCE as the container's main process, does its work, and
exits -- there is no persistent HTTP server, no exposed port dependency,
and no second implementation of the canonical CutSell pipeline. Exactly
like `pod_job_server.py`, the ONLY transport-specific code here is this
module; every actual editorial/pipeline call goes through the exact same
`serverless_handler.run_op(op, payload)` RunPod Serverless already uses.

Sequence (all in one process, one container boot -- "same Pod" per the
standing directive, since a Pod's dockerStartCmd cannot be re-invoked
with a second command once the container is running):
  1. Parse the job payload from CUTSELL_BENCHMARK_PAYLOAD_JSON (mirrors
     the exact shape of a Serverless job's `job["input"]`).
  2. Run tiny, non-editorial sanity checks (CUDA availability, ffmpeg,
     that the `cutsell_worker` package imports) and upload their result
     to a known S3 key -- this doubles as both the "prove direct
     execution works" validation step AND the orchestrator's runtime-
     readiness signal (S3-polled, never HTTP-polled).
  3. If sanity checks pass, call `run_op(op, payload)` directly -- the
     exact same canonical pipeline dispatch Serverless uses. `_focused`/
     `_locked_selection` already upload the full result.json and preview
     MP4 to S3 internally; this entrypoint additionally uploads run_op's
     own (compact) return value to a known key, so the orchestrator has
     an S3-native equivalent of a Serverless job's `.output` to poll for
     and read, without needing any container-log access.
  4. Any exception is caught and reported to a known S3 key rather than
     crashing silently -- container logs are not reliably reachable from
     the orchestrator (see D-042's open RunPod logs-access gap), so a
     terminal outcome must always be observable from S3 alone.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from .serverless_handler import _upload_artifact, run_op

_WORK_DIR = Path("/tmp/cutsell-pod-direct")
_SUBPROCESS_TIMEOUT_S = 120


def _run_capture(cmd: list[str], timeout: int = _SUBPROCESS_TIMEOUT_S) -> tuple[bool, str]:
    """Runs `cmd`, never raising -- a missing binary or a timeout is a
    failed check, not a crashed entrypoint."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, output.strip()
    except Exception as exc:  # noqa: BLE001 -- a failed sanity check, not a crash
        return False, f"exception running {cmd!r}: {exc}"


def run_sanity_checks() -> dict:
    """Direct, non-HTTP proof the container can actually do the work --
    NOT the pod_job_server /health endpoint. Three checks, all cheap:
    CUDA availability via torch, ffmpeg present, and the cutsell_worker
    package importable."""
    torch_ok, torch_out = _run_capture(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')",
        ]
    )
    cuda_available = False
    device_name = None
    if torch_ok:
        lines = [line.strip() for line in torch_out.splitlines() if line.strip()]
        if lines:
            cuda_available = lines[0].strip().lower() == "true"
        if len(lines) > 1:
            device_name = lines[1]

    ffmpeg_ok, ffmpeg_out = _run_capture(["ffmpeg", "-version"])
    import_ok, import_out = _run_capture(
        [sys.executable, "-c", "import cutsell_worker; print('cutsell_worker import ok')"]
    )

    return {
        "ok": bool(torch_ok and cuda_available and ffmpeg_ok and import_ok),
        "torch_check_ok": torch_ok,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "torch_check_output": torch_out[:2000],
        "ffmpeg_check_ok": ffmpeg_ok,
        "ffmpeg_check_output": (ffmpeg_out.splitlines()[0] if ffmpeg_ok and ffmpeg_out else ffmpeg_out[:500]),
        "cutsell_import_ok": import_ok,
        "cutsell_import_output": import_out[:500],
    }


def _write_and_upload_json(data: dict, prefix: str, filename: str) -> str:
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _WORK_DIR / filename
    local_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return _upload_artifact(str(local_path), key=f"{prefix}/{filename}", content_type="application/json")


def main() -> int:
    payload_raw = os.environ.get("CUTSELL_BENCHMARK_PAYLOAD_JSON") or ""
    try:
        payload = json.loads(payload_raw)
    except ValueError as exc:
        print(f"CUTSELL_BENCHMARK_PAYLOAD_JSON is not valid JSON: {exc}", flush=True)
        return 1
    if not isinstance(payload, dict):
        print("CUTSELL_BENCHMARK_PAYLOAD_JSON must decode to a JSON object", flush=True)
        return 1

    benchmark_id = str(payload.get("benchmark_id") or "").strip()
    if not benchmark_id:
        print("payload.benchmark_id is required -- refusing to guess an S3 key", flush=True)
        return 1
    op = str(payload.get("op") or "focused")
    prefix = f"cutsell/serverless/{benchmark_id}"

    print(f"--- [pod-direct] sanity checks for benchmark_id={benchmark_id} ---", flush=True)
    sanity = run_sanity_checks()
    _write_and_upload_json(sanity, prefix, "sanity_check.json")
    print(json.dumps(sanity, indent=2), flush=True)
    if not sanity["ok"]:
        print("--- [pod-direct] sanity checks FAILED -- not running the benchmark ---", flush=True)
        return 1

    print(f"--- [pod-direct] sanity checks passed; running op={op} ---", flush=True)
    try:
        result = run_op(op, payload)
    except Exception as exc:  # noqa: BLE001 -- must always report a terminal outcome to S3
        error_report = {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
        _write_and_upload_json(error_report, prefix, "pod-execution-error.json")
        print("--- [pod-direct] benchmark raised an exception ---", flush=True)
        print(error_report["traceback"], flush=True)
        return 1

    _write_and_upload_json(result, prefix, "run_output.json")
    print("--- [pod-direct] run_op output ---", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
