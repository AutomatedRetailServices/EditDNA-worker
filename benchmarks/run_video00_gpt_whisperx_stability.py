"""Bounded QA transport: five independent calls to the existing full editor.

No editorial decisions live here. One private configuration snapshot is shared
by all trials; each trial starts a new Modal CLI/app and a fresh ASR provider.
An attempted trial cannot be replayed, even when it failed. QA failures are
recorded independently and never trigger another paid invocation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TRIAL_COUNT = 5
SOURCE_KEY = "Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4"
SOURCE_SHA256 = "b37059b1790cf3eb0447bb54595cc99f6668aadc5f65dafa9cdf6181621ef9f5"
PROVIDER = os.environ.get("CUTSELL_STABILITY_PROVIDER", "gpt-transcribe-whisperx")
if PROVIDER not in {"gpt-transcribe-whisperx", "deepgram-nova-3-multi"}:
    raise ValueError("Unsupported stability provider")
TRIAL_TAG = "deepgram" if PROVIDER == "deepgram-nova-3-multi" else "gptwx"
import runpy

# Read the dependency-free profile without executing cutsell_worker/__init__.py
# in the CPU transport environment. The GPU image owns inference dependencies.
WATCH_LISTEN_CAPABILITIES = runpy.run_path(str(ROOT / "cutsell_worker/watch_listen_runtime.py"))["DEPENDENCIES"]

OVERLAYS = {
    "CUTSELL_WATCH_LISTEN_AUTOMATIC": "1",
    **{"CUTSELL_" + name: "1" for name in WATCH_LISTEN_CAPABILITIES},
    "CUTSELL_HYBRID_LLM_ENABLED": "1",
    "CUTSELL_HYBRID_PROVIDER": "google",
    "CUTSELL_UNIFIED_REALIZATION_RESOLVER": "AUTHORITATIVE",
    "CUTSELL_ASR_CANONICAL_NORMALIZATION": "1",
    "CUTSELL_SEMANTIC_COMPUTE_PLANNER": "1",
    "CUTSELL_VALIDATION_ASR_PROVIDER": PROVIDER,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_dir() -> Path:
    return Path(os.environ["RUNNER_TEMP"]) / "cutsell-five-private"


def claim_trial(private: Path, index: int) -> None:
    if index not in range(1, TRIAL_COUNT + 1):
        raise ValueError("Only the five authorized trials may run")
    if (private / "STOP").exists():
        raise RuntimeError("An earlier invocation has an uncertain terminal state")
    for earlier in range(1, index):
        if not (private / f"trial-{earlier}.terminal").exists():
            raise RuntimeError("Trials must finish sequentially before the next starts")
    # Atomic, irreversible claim before dispatch; failed attempts also consume it.
    with (private / f"trial-{index}.claimed").open("x") as handle:
        handle.write("No automatic paid retry\n")


def s3_client(env: dict):
    import boto3
    return boto3.client(
        "s3", region_name=env.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=env.get("AWS_SESSION_TOKEN"),
    )


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise RuntimeError("Missing CPU media tools before paid dispatch: " + ", ".join(missing))


def prepare() -> None:
    require_media_tools()
    import requests
    from cutsell_worker.active_path_identity import package_fingerprint
    from modal_gpu_config import require_modal_token_env

    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise RuntimeError("Paid batch reruns are disabled")
    require_modal_token_env(os.environ)
    private = private_dir()
    private.mkdir(mode=0o700, exist_ok=False)
    response = requests.get(
        "https://rest.runpod.io/v1/templates",
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
        timeout=(15, 60),
    )
    if response.status_code != 200:
        raise RuntimeError(f"Template preflight HTTP {response.status_code}")
    matches = [x for x in response.json() if x.get("name") == "EditDNA-Worker-2"]
    if len(matches) != 1:
        raise RuntimeError("Expected one canonical worker template")
    env = {str(k): str(v) for k, v in matches[0]["env"].items()}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env.update(OVERLAYS, CUTSELL_BUILD_GIT_SHA=head)
    key = env.get("DEEPGRAM_API_KEY" if TRIAL_TAG == "deepgram" else "OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-admin-"):
        raise RuntimeError("Required inference key missing; no GPU started")
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET"):
        if not env.get(name):
            raise RuntimeError(f"Missing required environment variable: {name}")
    # Mask credential-like values of any length; long config values too.
    for name, value in env.items():
        sensitive = any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD", "BUCKET"))
        if value and (sensitive or len(value) >= 12) and name not in OVERLAYS and name != "CUTSELL_BUILD_GIT_SHA":
            escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::add-mask::{escaped}")
    config = private / "environment.json"
    write_json(config, env)
    config.chmod(0o600)
    obj = s3_client(env).head_object(Bucket=env["S3_BUCKET"], Key=SOURCE_KEY)
    if not obj.get("ContentLength"):
        raise RuntimeError("Source preflight returned an empty object")
    lock = {"config_sha256": sha256(config), "head": head,
            "package": package_fingerprint(ROOT / "cutsell_worker"),
            "source_etag": obj.get("ETag"), "source_bytes": obj["ContentLength"]}
    write_json(private / "lock.json", lock)
    output = ROOT / "stability-artifacts"
    output.mkdir(exist_ok=True)
    write_json(output / "batch.json", {
        "schema": "cutsell.asr_provider_stability.v1", "authorized_trials": TRIAL_COUNT,
        "run_id": os.environ["GITHUB_RUN_ID"], "build_sha": head,
        "package": lock["package"], "source_key": SOURCE_KEY,
        "expected_source_sha256": SOURCE_SHA256, "overlays": OVERLAYS,
        "configuration_snapshot_count": 1, "cross_trial_transcript_reuse": False,
        "retries": 0, "gpu": "L4", "remote_timeout_sec": 5400,
        "auto_speech_visual_microtrim": True,
    })
    print("Preflight passed: one private configuration snapshot, five sequential independent trials")


def collect(compact: dict, env: dict, output: Path, index: int) -> dict:
    from benchmarks.validate_video00_regression_qa import validate
    client = s3_client(env)
    downloads = [("result_uri", "result.json"),
                 ("preview_uri", f"Video00_{TRIAL_TAG}_Prueba_{index}.mp4"),
                 ("diagnostic_preview_uri", f"Video00_Prueba_{index}_INVALIDADO.mp4")]
    for field, filename in downloads:
        if compact.get(field):
            uri = urlparse(compact[field])
            if uri.scheme != "s3" or uri.netloc != env["S3_BUCKET"]:
                raise RuntimeError("Unexpected artifact storage location")
            client.download_file(uri.netloc, uri.path.lstrip("/"), str(output / filename))
    result_path = output / "result.json"
    if not result_path.exists():
        return {"engine_ok": False, "error_type": compact.get("error_type"),
                "terminal_state": compact.get("terminal_state"), "result_present": False}
    result = json.loads(result_path.read_text())
    reports = {}
    for name, manifest in (("editorial", "video00_editorial_acceptance.json"),
                           ("gold", "video00_regression_qa.json")):
        _, report = validate(str(result_path), str(ROOT / "benchmarks" / manifest))
        write_json(output / f"{name}-qa.json", report)
        reports[name] = {k: report.get(k) for k in ("qa_pass", "passed_check_count", "failed_check_count")}
    for media in output.glob("*.mp4"):
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                "-of", "json", str(media)], capture_output=True, text=True, check=True)
        write_json(output / "media-probe.json", json.loads(probe.stdout))
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(media),
                                  "-f", "null", "-"], capture_output=True, text=True)
        reports["media"] = {"filename": media.name, "sha256": sha256(media),
                            "bytes": media.stat().st_size, "decode_ok": decoded.returncode == 0,
                            "decode_errors": decoded.stderr[:2000]}
    audit = result.get("asr_provider_audit") or {}
    identity = result.get("active_path_identity") or {}
    lock = json.loads((private_dir() / "lock.json").read_text())
    return {"engine_ok": bool(compact.get("ok")), "result_present": True,
            "source_sha256": result.get("source_media_sha256"),
            "source_match": result.get("source_media_sha256") == SOURCE_SHA256,
            "worker_sha_match": identity.get("build_git_sha") == lock["head"],
            "package_match": (identity.get("package") or {}).get("sha256") == lock["package"]["sha256"],
            "asr_provider": audit.get("provider"), "asr_config": audit.get("config_fingerprint"),
            "words": audit.get("word_count"), "chunks": len(audit.get("chunks") or []),
            "cache_hit_count": audit.get("cache_hit_count"),
            "engine_elapsed_sec": result.get("elapsed_sec"),
            "output_duration_sec": result.get("output_duration_sec"),
            "selected_count": result.get("selected_count"),
            "technical_qc": (result.get("live_render_qc") or {}).get("status"),
            "delivery_status": result.get("delivery_status"), "qa": reports}


def run_trial(index: int) -> None:
    private = private_dir()
    lock = json.loads((private / "lock.json").read_text())
    config = private / "environment.json"
    if sha256(config) != lock["config_sha256"]:
        raise RuntimeError("Configuration changed between trials")
    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise RuntimeError("Paid batch reruns are disabled")
    claim_trial(private, index)
    output = ROOT / "stability-artifacts" / f"trial-{index}"
    output.mkdir(exist_ok=False)
    benchmark_id = f"video00-{TRIAL_TAG}-five-{os.environ['GITHUB_RUN_ID']}-{index}"
    env = json.loads(config.read_text())
    child_env = dict(os.environ)
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION", "S3_BUCKET"):
        if key in env:
            child_env[key] = env[key]
    child_env.update(CUTSELL_ENV_JSON_PATH=str(config), CUTSELL_VALIDATION_ASR_PROVIDER=PROVIDER,
                     CUTSELL_BENCHMARK_PAYLOAD_JSON=json.dumps({"op": "focused", "source_key": SOURCE_KEY,
                         "benchmark_id": benchmark_id, "auto_speech_visual_microtrim": True}))
    compact_path = ROOT / "modal-video00-result.json"
    if compact_path.exists():
        raise RuntimeError("Uncollected result exists before a fresh trial")
    summary = {"trial": index, "benchmark_id": benchmark_id, "configuration_unchanged": True,
               "started_at_epoch": time.time(), "status": "started"}
    write_json(output / "summary.json", summary)
    print(f"Starting authorized trial {index}/{TRIAL_COUNT}: {benchmark_id}", flush=True)
    try:
        with (output / "modal.log").open("w") as log:
            completed = subprocess.run(["modal", "run", "modal_video00_full_benchmark.py"],
                                       cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        summary["modal_exit_code"] = completed.returncode
        if completed.returncode != 0:
            (private / "STOP").touch()
            raise RuntimeError("Modal CLI failed; remaining trials blocked to prevent overlap")
        if not compact_path.exists():
            (private / "STOP").touch()
            raise RuntimeError("Missing terminal compact result")
        compact_path.replace(output / "compact.json")
        compact = json.loads((output / "compact.json").read_text())
        if compact.get("terminal_state") == "local_wrapper_exception":
            (private / "STOP").touch()
        # The blocking Modal process has exited and its ephemeral app stopped.
        (private / f"trial-{index}.terminal").touch()
        summary.update(collect(compact, env, output, index))
        summary["status"] = "completed" if summary.get("engine_ok") else "engine_failed"
        if summary.get("source_match") is False:
            (private / "STOP").touch()
    except Exception as exc:
        summary.update(status="failed", harness_error_type=type(exc).__name__)
        raise
    finally:
        summary["wall_elapsed_sec"] = round(time.time() - summary["started_at_epoch"], 3)
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def finalize() -> int:
    output = ROOT / "stability-artifacts"
    rows = []
    for index in range(1, TRIAL_COUNT + 1):
        path = output / f"trial-{index}" / "summary.json"
        rows.append(json.loads(path.read_text()) if path.exists() else {"trial": index, "status": "not_run"})
    complete = all(r.get("engine_ok") and r.get("source_match") and r.get("worker_sha_match")
                   and r.get("package_match") for r in rows)
    accepted = complete and all((r.get("qa") or {}).get("editorial", {}).get("qa_pass")
                               and r.get("technical_qc") == "PASS" for r in rows)
    write_json(output / "five-trial-summary.json", {"trials": rows, "all_engine_runs_completed": complete,
                                                   "all_editorially_accepted": accepted})
    print(json.dumps({"all_engine_runs_completed": complete, "all_editorially_accepted": accepted}))
    return 0 if accepted else 1


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "prepare":
        prepare()
    elif command == "trial":
        run_trial(int(sys.argv[2]))
    elif command == "finalize":
        raise SystemExit(finalize())
    else:
        raise SystemExit("Expected prepare, trial INDEX, or finalize")
