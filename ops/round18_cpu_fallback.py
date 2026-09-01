from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import requests

RUNPOD = "https://rest.runpod.io/v1"
SOURCE_SHA = os.environ["SOURCE_SHA"]
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
TEMPLATE_NAME = os.environ.get("RUNPOD_TEMPLATE_NAME", "EditDNA-Worker-2")
EXACT_SOURCE = Path("exact-source").resolve()


def runpod(method: str, path: str, **kwargs):
    return requests.request(method, f"{RUNPOD}{path}", headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}, timeout=45, **kwargs)


def clean_stale_pods() -> None:
    r = runpod("GET", "/pods")
    r.raise_for_status()
    for pod in r.json():
        name = str(pod.get("name") or "")
        if name.startswith("cutsell-staging-worker"):
            pid = str(pod.get("id") or "")
            if not pid:
                continue
            d = runpod("DELETE", f"/pods/{pid}")
            print(f"cpu_fallback_cleanup pod={pid} http={d.status_code}", flush=True)
            if d.status_code not in (204, 404):
                raise RuntimeError(f"failed to delete stale pod {pid}: {d.status_code}")


def load_template_env() -> dict[str, str]:
    r = runpod("GET", "/templates")
    r.raise_for_status()
    template = next((x for x in r.json() if x.get("name") == TEMPLATE_NAME), None)
    if not isinstance(template, dict):
        raise RuntimeError(f"template not found: {TEMPLATE_NAME}")
    allowed = {
        "ASR_DEVICE", "ASR_ENABLED", "BAD_TAKES_ENABLED", "BOUNDARY_REFINER_ENABLED",
        "BOUNDARY_REFINER_HEAD_STEP_SEC", "BOUNDARY_REFINER_MIN_DURATION_SEC",
        "BOUNDARY_REFINER_TAIL_STEP_SEC", "FFMPEG_BIN", "FFPROBE_BIN", "HEAD_TRIM_SEC",
        "PRESIGN_EXPIRES", "PYTHONPATH", "S3_ACL", "S3_BUCKET", "S3_PREFIX", "TAIL_TRIM_SEC",
        "VISION_ENABLED", "VISION_INTERVAL_SEC", "VISION_MAX_SAMPLES", "VISUAL_BAD_THRESHOLD",
        "WHISPER_DEVICE", "WHISPER_MODEL", "W_FACE", "W_VISION", "W_VISUAL",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
    }
    raw = template.get("env") or {}
    env = {str(k): str(v) for k, v in raw.items() if k in allowed and v not in (None, "")}
    for required in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET"):
        if not env.get(required):
            raise RuntimeError(f"template missing {required}")
    return env


def configure_cpu_runtime(template_env: dict[str, str]) -> None:
    os.environ.update(template_env)
    os.environ.update({
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_EDITORIAL_MODE": "clean_cut",
        "CUTSELL_ASR_MODEL": "medium",
        "CUTSELL_CLEAN_CUT_JUDGE": "0",
        "CUTSELL_HYBRID_PROVIDER": "google",
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PRIMARY_MODEL": "gemini-3.5-flash-lite",
        "CUTSELL_HYBRID_ESCALATION_MODEL": "gemini-3.6-flash",
        "CUTSELL_HYBRID_MAX_SESSION_USD": "0.0075",
        "CUTSELL_HYBRID_MAX_EDIT_USD": "0.50",
        "CUTSELL_HYBRID_TEST_BUDGET_USD": "0.50",
        "PYTHONPATH": str(EXACT_SOURCE),
    })
    # These legacy/external paths must stay dormant.
    for bad in ("OPENAI_API_KEY", "EDITDNA_USE_LLM", "EDITDNA_LLM_MODEL"):
        os.environ.pop(bad, None)
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError("GEMINI_API_KEY missing")


def load_ops_helper():
    path = Path("ops/round14_capacity_failover.py").resolve()
    spec = importlib.util.spec_from_file_location("round14_cpu_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not (EXACT_SOURCE / "cutsell_worker").is_dir():
        raise RuntimeError("exact-source/cutsell_worker missing")
    clean_stale_pods()
    template_env = load_template_env()
    configure_cpu_runtime(template_env)
    sys.path.insert(0, str(EXACT_SOURCE))

    from cutsell_worker.focused_validation_job import run_focused_clean_cut_benchmark

    keys = [
        "Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4",
        "Editdna longform validation/VIDEO-2026-07-30-09-24-13.mp4",
        "Editdna longform validation/VIDEO-2026-07-30-10-22-46.mp4",
    ]
    payload = {
        "benchmark_id": f"focused-round18-cpu-{RUN_ID}-{SOURCE_SHA[:12]}",
        "source_prefix": "Editdna longform validation/",
        "source_keys": keys,
        "expected_external_brain_calls_enabled": True,
    }
    started = time.monotonic()
    print(json.dumps({"event": "cpu_benchmark_start", "source_sha": SOURCE_SHA, "sources": keys}), flush=True)
    result = run_focused_clean_cut_benchmark(payload)
    elapsed = round(time.monotonic() - started, 3)
    json.dump(result, open("focused-result.json", "w"), ensure_ascii=False, indent=2)
    if result.get("source_count") != 3 or result.get("completed_count") != 3:
        raise RuntimeError(f"CPU focused benchmark incomplete: {result}")
    if result.get("execution_failure_count") or result.get("provider_failure_count"):
        raise RuntimeError(f"CPU focused benchmark reported failures: {result}")

    helper = load_ops_helper()
    report = helper.download_artifacts(result, template_env)
    helper.validate_gold(report)
    status = {
        "status": "gold_pass",
        "execution_mode": "github_actions_cpu",
        "source_sha": SOURCE_SHA,
        "run_id": RUN_ID,
        "elapsed_sec": elapsed,
        "completed_count": report.get("completed_count"),
    }
    json.dump(status, open("round18-status.json", "w"), ensure_ascii=False, indent=2)
    print("ROUND18_CPU_GOLD_PASS=true", flush=True)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
