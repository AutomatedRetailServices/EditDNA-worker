from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
from redis import Redis
from rq import Queue, Worker
from rq.job import Job

RUNPOD = "https://rest.runpod.io/v1"
POD_PREFIX = "cutsell-staging-worker"
SOURCE_SHA = os.environ["SOURCE_SHA"]
IMAGE_DIGEST = os.environ["IMAGE_DIGEST"]
IMMUTABLE_IMAGE = os.environ["IMMUTABLE_IMAGE"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
REDIS_URL = os.environ["REDIS_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TEMPLATE_NAME = os.environ.get("RUNPOD_TEMPLATE_NAME", "EditDNA-Worker-2")
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {RUNPOD_API_KEY}"})
redis = Redis.from_url(REDIS_URL)


def api(method: str, path: str, **kwargs):
    r = session.request(method, f"{RUNPOD}{path}", timeout=45, **kwargs)
    return r


def delete_pod(pod_id: str) -> None:
    try:
        r = api("DELETE", f"/pods/{pod_id}")
        print(f"delete pod={pod_id} http={r.status_code}", flush=True)
    except Exception as exc:
        print(f"delete pod={pod_id} exception={type(exc).__name__}", flush=True)
    time.sleep(2)


def clean_stale() -> None:
    r = api("GET", "/pods")
    r.raise_for_status()
    for pod in r.json():
        name = str(pod.get("name") or "")
        if name.startswith(POD_PREFIX):
            pid = str(pod.get("id") or "")
            if pid:
                print(f"clean_stale pod={pid} name={name}", flush=True)
                delete_pod(pid)


def safe_template_env() -> tuple[dict, str]:
    r = api("GET", "/templates")
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
    env = {k: v for k, v in raw.items() if k in allowed and v not in (None, "")}
    for required in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET"):
        if not env.get(required):
            raise RuntimeError(f"template missing {required}")
    mount = str(template.get("volumeMountPath") or "/workspace")
    return env, mount


def build_env(base: dict) -> dict:
    env = dict(base)
    env.update({
        "REDIS_URL": REDIS_URL,
        "GEMINI_API_KEY": GEMINI_API_KEY,
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
    })
    for bad in ("OPENAI_API_KEY", "EDITDNA_USE_LLM", "EDITDNA_LLM_MODEL"):
        env.pop(bad, None)
    return env


def find_worker(base_env: dict, mount: str) -> tuple[str, str, str, str, list[dict]]:
    before = {w.name for w in Worker.all(connection=redis)}
    candidates = [
        ("COMMUNITY", "NVIDIA A40"),
        ("COMMUNITY", "NVIDIA RTX A5000"),
        ("COMMUNITY", "NVIDIA RTX A6000"),
        ("COMMUNITY", "NVIDIA L4"),
        ("COMMUNITY", "NVIDIA RTX A4000"),
        ("COMMUNITY", "NVIDIA GeForce RTX 3090"),
        ("COMMUNITY", "NVIDIA GeForce RTX 4090"),
        ("SECURE", "NVIDIA A40"),
        ("SECURE", "NVIDIA RTX A6000"),
        ("SECURE", "NVIDIA L4"),
        ("SECURE", "NVIDIA GeForce RTX 4090"),
    ]
    env = build_env(base_env)
    attempts: list[dict] = []
    for cloud, gpu in candidates:
        payload = {
            "name": POD_PREFIX,
            "imageName": IMMUTABLE_IMAGE,
            "cloudType": cloud,
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": [gpu],
            "gpuTypePriority": "custom",
            "interruptible": False,
            "containerDiskInGb": 30,
            "volumeInGb": 20,
            "volumeMountPath": mount,
            "env": env,
        }
        r = api("POST", "/pods", json=payload)
        if r.status_code != 201:
            attempts.append({"cloud": cloud, "gpu": gpu, "result": "create_rejected", "http": r.status_code})
            print(f"create_rejected cloud={cloud} gpu={gpu} http={r.status_code}", flush=True)
            continue
        info = r.json()
        pid = str(info.get("id") or "")
        created = time.time()
        print(json.dumps({"event": "pod_created", "pod": pid, "cloud": cloud, "gpu": gpu, "cost": info.get("costPerHr")}), flush=True)
        runtime_seen = False
        runtime_deadline = time.time() + 120
        while time.time() < runtime_deadline:
            state = api("GET", f"/pods/{pid}")
            if state.status_code == 200:
                data = state.json()
                runtime_seen = isinstance(data.get("runtime"), dict)
                print(json.dumps({"event": "pod_state", "pod": pid, "cloud": cloud, "gpu": gpu, "desired": data.get("desiredStatus"), "runtime": runtime_seen, "last": data.get("lastStatusChange")}), flush=True)
                if runtime_seen:
                    break
            time.sleep(10)
        if not runtime_seen:
            attempts.append({"cloud": cloud, "gpu": gpu, "pod": pid, "result": "no_runtime"})
            delete_pod(pid)
            continue
        worker_deadline = time.time() + 360
        chosen_worker = None
        while time.time() < worker_deadline:
            now = datetime.now(timezone.utc)
            for worker in Worker.all(connection=redis):
                hb = worker.last_heartbeat
                if not hb or worker.name in before:
                    continue
                if hb.tzinfo is None:
                    hb = hb.replace(tzinfo=timezone.utc)
                if hb.timestamp() >= created - 5 and (now - hb).total_seconds() <= 45 and worker.get_state() in {"idle", "busy"}:
                    chosen_worker = worker.name
                    break
            if chosen_worker:
                break
            time.sleep(10)
        if chosen_worker:
            attempts.append({"cloud": cloud, "gpu": gpu, "pod": pid, "result": "fresh_worker", "worker": chosen_worker})
            return pid, cloud, gpu, chosen_worker, attempts
        attempts.append({"cloud": cloud, "gpu": gpu, "pod": pid, "result": "runtime_without_rq"})
        delete_pod(pid)
    raise RuntimeError("No GPU/cloud candidate produced a fresh RQ worker")


def enqueue_and_wait(cloud: str, gpu: str) -> dict:
    keys = [
        "Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4",
        "Editdna longform validation/VIDEO-2026-07-30-09-24-13.mp4",
        "Editdna longform validation/VIDEO-2026-07-30-10-22-46.mp4",
    ]
    payload = {
        "benchmark_id": f"focused-round14-{RUN_ID}-{SOURCE_SHA[:12]}",
        "source_prefix": "Editdna longform validation/",
        "source_keys": keys,
        "expected_external_brain_calls_enabled": True,
    }
    job = Queue("cutsell", connection=redis).enqueue(
        "cutsell_worker.focused_validation_job.run_focused_clean_cut_benchmark",
        payload,
        job_timeout=6600,
        result_ttl=86400,
        failure_ttl=86400,
        meta={
            "dataset_role": "clean_cut_gold_raw_focused",
            "source_keys": keys,
            "worker_source_sha": SOURCE_SHA,
            "worker_image_digest": IMAGE_DIGEST,
            "chosen_cloud": cloud,
            "chosen_gpu": gpu,
        },
    )
    print(f"job_id={job.id}", flush=True)
    deadline = time.monotonic() + 6600
    last = None
    while time.monotonic() < deadline:
        status = getattr(job.get_status(refresh=True), "value", str(job.get_status(refresh=True)))
        meta = dict(job.meta or {})
        snap = (status, meta.get("stage"), meta.get("progress_percent"), meta.get("current_source"))
        if snap != last:
            print(json.dumps({"status": status, "stage": meta.get("stage"), "progress": meta.get("progress_percent"), "source": meta.get("current_source")}, ensure_ascii=False), flush=True)
            last = snap
        if status == "finished":
            result = job.result
            if not isinstance(result, dict):
                raise RuntimeError("benchmark returned non-dict result")
            return result
        if status in {"failed", "stopped", "canceled"}:
            raise RuntimeError(job.exc_info or f"benchmark status={status}")
        time.sleep(15)
    raise TimeoutError("benchmark timed out")


def download_artifacts(result: dict, base_env: dict) -> dict:
    Path("focused-previews").mkdir(exist_ok=True)
    s3 = boto3.client(
        "s3",
        region_name=base_env.get("AWS_REGION") or "us-east-1",
        aws_access_key_id=base_env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=base_env["AWS_SECRET_ACCESS_KEY"],
    )
    def parse(uri: str):
        u = urlparse(uri)
        return u.netloc, u.path.lstrip("/")
    b, k = parse(result["report_uri"])
    s3.download_file(b, k, "cutsell-focused-clean-cut-benchmark.json")
    for uri in result.get("preview_uris") or []:
        b, k = parse(uri)
        s3.download_file(b, k, str(Path("focused-previews") / Path(k).name))
    report = json.load(open("cutsell-focused-clean-cut-benchmark.json"))
    if report.get("source_count") != 3 or report.get("completed_count") != 3 or report.get("execution_failure_count") or report.get("provider_failure_count"):
        raise RuntimeError("invalid focused report")
    if len(list(Path("focused-previews").glob("*.mp4"))) != 3:
        raise RuntimeError("expected exactly 3 previews")
    return report


def rows(result: dict, key: str):
    out = []
    for item in result.get(key) or result.get(key + "_takes") or []:
        if not isinstance(item, dict):
            continue
        s = item.get("start", item.get("source_start")); e = item.get("end", item.get("source_end"))
        if s is not None and e is not None:
            out.append((float(s), float(e), str(item.get("text") or "")))
    return out


def validate_gold(report: dict) -> None:
    results = report.get("results") or []
    v00 = next((x for x in results if str(x.get("source_key") or "").endswith("VIDEO-2026-07-30-09-18-03.mp4")), None)
    v02 = next((x for x in results if str(x.get("source_key") or "").endswith("VIDEO-2026-07-30-09-24-13.mp4")), None)
    v03 = next((x for x in results if str(x.get("source_key") or "").endswith("VIDEO-2026-07-30-10-22-46.mp4")), None)
    if not all(isinstance(x, dict) for x in (v00, v02, v03)):
        raise RuntimeError("missing focused result")
    low = lambda x: " ".join(str(x or "").casefold().split())
    ss, s02, s03 = rows(v00, "selected"), rows(v02, "selected"), rows(v03, "selected")
    json.dump({"video00_selected": ss, "video02_selected": s02, "video03_selected": s03}, open("gold-selected.json", "w"), ensure_ascii=False, indent=2)
    if any(s < 116.9 and e > 108.2 for s, e, _ in ss): raise RuntimeError("broken 108-116 retry selected")
    if not any(119.5 <= s <= 122.0 and e >= 123.5 and "sonograf" in low(t) for s, e, t in ss): raise RuntimeError("clean 120 retry missing")
    if any("mínimo dos estados" in low(t) for _, _, t in ss): raise RuntimeError("broken thyroid attempt selected")
    if not any(34 <= s <= 37 and e >= 44 and "nunca se nos ocurrió" in low(t) for s, e, t in ss): raise RuntimeError("clean thyroid retake missing")
    if any(235 <= s <= 237.5 and e >= 243 for s, e, _ in ss): raise RuntimeError("incomplete stomach attempt selected")
    if not any(257.5 <= s <= 259.5 and e >= 268 and ("gastritis" in low(t) or "digestión" in low(t) or "digestion" in low(t)) for s, e, t in ss): raise RuntimeError("corrected stomach retry missing")
    if not any(294.5 <= s <= 296.5 and e >= 312.5 for s, e, _ in ss): raise RuntimeError("295 close missing")
    if any(318.5 <= s <= 321.5 and e >= 333 for s, e, _ in ss): raise RuntimeError("319 prefix selected")
    if any(334.5 <= s <= 337 and e >= 345 for s, e, _ in ss): raise RuntimeError("335 continuation selected")
    if any(low(t).endswith("de los") for _, _, t in ss): raise RuntimeError("open de-los selected")
    if not any(355 <= s <= 357.5 and e >= 361.3 and ("haz ejercicio" in low(t) or "haces ejercicio" in low(t)) for s, e, t in ss): raise RuntimeError("complete CTA missing")
    if any(108.5 <= s <= 111.5 and e <= 112.5 for s, e, _ in s02) or any("trying to say in character" in low(t) for _, _, t in s02): raise RuntimeError("Video02 BTS fragment selected")
    text03 = " ".join(low(t) for _, _, t in s03)
    if "la barrera cutánea" not in text03: raise RuntimeError("Video03 lost la barrera cutánea")
    if "te la te hace como" in text03: raise RuntimeError("Video03 failed suffix resurfaced")


def main() -> None:
    clean_stale()
    base_env, mount = safe_template_env()
    chosen_pid = None
    attempts = []
    status = {"source_sha": SOURCE_SHA, "image_digest": IMAGE_DIGEST, "run_id": RUN_ID, "status": "starting"}
    try:
        pid, cloud, gpu, worker, attempts = find_worker(base_env, mount)
        chosen_pid = pid
        Path("pod-id.txt").write_text(pid)
        status.update({"status": "worker_live", "pod_id": pid, "cloud": cloud, "gpu": gpu, "worker": worker})
        json.dump(status, open("round14-status.json", "w"), indent=2)
        result = enqueue_and_wait(cloud, gpu)
        json.dump(result, open("focused-result.json", "w"), ensure_ascii=False, indent=2)
        report = download_artifacts(result, base_env)
        validate_gold(report)
        status.update({"status": "gold_pass", "completed_count": report.get("completed_count")})
        json.dump(status, open("round14-status.json", "w"), indent=2)
        print("ROUND14_GOLD_PASS=true", flush=True)
    finally:
        json.dump(attempts, open("capacity-attempts.json", "w"), indent=2)
        if chosen_pid:
            delete_pod(chosen_pid)


if __name__ == "__main__":
    main()
