"""Pure, modal-package-free diagnostic logic for the Modal GPU minimal
smoke test (D-043). Deliberately separated from `modal_gpu_minimal_test.py`
(which defines the actual Modal App/Function and does require the `modal`
package) so this logic is directly unit-testable without modal installed,
the same "no editor implementation duplicated" precedent as
`cutsell_worker.pod_direct_benchmark_entrypoint.run_sanity_checks`.

This module runs INSIDE the Modal container (imported by the decorated
function in `modal_gpu_minimal_test.py`) -- it never runs the CutSell
editorial pipeline, only reports environment facts: GPU model, CUDA
availability/version, torch version, compute capability, ffmpeg/ffprobe
presence, Python version, and elapsed runtime.
"""
from __future__ import annotations

import subprocess
import sys
import time


def _run_capture(cmd: list[str], timeout: float = 30.0) -> tuple[bool, str]:
    """Runs `cmd`, returns (succeeded, combined stdout+stderr). Never
    raises -- a missing binary or timeout is reported as a failure, not
    an exception that would crash the whole diagnostic."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
    except Exception as exc:  # noqa: BLE001 -- any failure here is itself the diagnostic result
        return False, str(exc)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def collect_gpu_diagnostics() -> dict:
    """Returns a fully JSON-serializable dict reporting every field D-043's
    live gate requires. `ok` is True only when CUDA is available AND both
    ffmpeg/ffprobe are present -- a smoke test that can't see the GPU or is
    missing tooling is a real failure, not a partial pass."""
    start = time.monotonic()
    result: dict = {
        "python_version": sys.version.split()[0],
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_model": None,
        "compute_capability": None,
        "torch_error": None,
    }

    try:
        import torch  # local import: only ever available inside the Modal container

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_version"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpu_model"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            result["compute_capability"] = f"{major}.{minor}"
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole smoke test
        result["torch_error"] = str(exc)

    ffmpeg_ok, ffmpeg_out = _run_capture(["ffmpeg", "-version"])
    result["ffmpeg_available"] = ffmpeg_ok
    result["ffmpeg_version"] = _first_line(ffmpeg_out) if ffmpeg_ok else None

    ffprobe_ok, ffprobe_out = _run_capture(["ffprobe", "-version"])
    result["ffprobe_available"] = ffprobe_ok
    result["ffprobe_version"] = _first_line(ffprobe_out) if ffprobe_ok else None

    result["ok"] = bool(result["cuda_available"] and ffmpeg_ok and ffprobe_ok)
    result["completion_status"] = "COMPLETED" if result["ok"] else "COMPLETED_WITH_ISSUES"
    result["elapsed_s"] = time.monotonic() - start
    return result
