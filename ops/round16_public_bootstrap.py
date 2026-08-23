from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from rq import Worker

import ops.round14_capacity_failover as base

PUBLIC_IMAGE = "madiator2011/better-pytorch:cuda12.4-torch2.6.0"
SOURCE_BUNDLE = Path("source-bundle.tgz")


def upload_source_bundle(base_env: dict) -> tuple[str, str, str]:
    if not SOURCE_BUNDLE.exists():
        raise RuntimeError("source-bundle.tgz missing")
    bucket = base_env["S3_BUCKET"]
    region = base_env.get("AWS_REGION") or "us-east-1"
    key = f"cutsell/bootstrap/round16/{base.RUN_ID}/{base.SOURCE_SHA}/source-bundle.tgz"
    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=base_env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=base_env["AWS_SECRET_ACCESS_KEY"],
    )
    s3.upload_file(str(SOURCE_BUNDLE), bucket, key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=10800,
    )
    print(json.dumps({"event": "source_bundle_uploaded", "bucket": bucket, "key": key, "bytes": SOURCE_BUNDLE.stat().st_size}), flush=True)
    return url, bucket, key


def bootstrap_command() -> str:
    # No GitHub or private registry credentials are required on the Pod. The exact
    # source archive is delivered by a short-lived S3 presigned URL.
    return r'''set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /app
cd /app
python - <<'PY'
import os, urllib.request
url=os.environ['CUTSELL_SOURCE_BUNDLE_URL']
urllib.request.urlretrieve(url, '/tmp/cutsell-source.tgz')
print('source_bundle_downloaded=true', flush=True)
PY
tar -xzf /tmp/cutsell-source.tgz -C /app
apt-get update
apt-get install -y --no-install-recommends build-essential python3-dev ffmpeg git curl pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
rm -rf /var/lib/apt/lists/*
python -m pip install --no-cache-dir -r /app/requirements.cutsell.worker.txt
python -m pip uninstall -y openai >/dev/null 2>&1 || true
export PYTHONPATH=/app
python - <<'PY'
import importlib.util, os
assert importlib.util.find_spec('cutsell_worker') is not None
assert importlib.util.find_spec('openai') is None
print('bootstrap_source_sha='+os.environ.get('CUTSELL_EXPECTED_SOURCE_SHA',''), flush=True)
print('bootstrap_ready=true', flush=True)
PY
exec python -m rq.cli worker -u "$REDIS_URL" --worker-ttl 14400 cutsell
'''


def find_public_worker(base_env: dict, mount: str, bundle_url: str):
    before = {w.name for w in Worker.all(connection=base.redis)}
    env = base.build_env(base_env)
    env.update({
        "CUTSELL_SOURCE_BUNDLE_URL": bundle_url,
        "CUTSELL_EXPECTED_SOURCE_SHA": base.SOURCE_SHA,
        "PYTHONPATH": "/app",
    })
    gpus = [
        "NVIDIA GeForce RTX 5090",
        "NVIDIA L40S",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA L40",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA A40",
        "NVIDIA RTX A6000",
        "NVIDIA L4",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 3090",
    ]
    attempts = []
    for cloud in ("SECURE", "COMMUNITY"):
        payload = {
            "name": base.POD_PREFIX,
            "imageName": PUBLIC_IMAGE,
            "cloudType": cloud,
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": gpus,
            "gpuTypePriority": "availability",
            "dataCenterPriority": "availability",
            "interruptible": False,
            "containerDiskInGb": 40,
            "volumeInGb": 20,
            "volumeMountPath": mount,
            "env": env,
            "dockerEntrypoint": ["bash", "-lc"],
            "dockerStartCmd": [bootstrap_command()],
        }
        r = base.api("POST", "/pods", json=payload)
        if r.status_code != 201:
            attempts.append({"cloud": cloud, "result": "create_rejected", "http": r.status_code, "body": r.text[:500]})
            print(json.dumps({"event": "public_create_rejected", "cloud": cloud, "http": r.status_code, "body": r.text[:300]}, ensure_ascii=False), flush=True)
            continue
        info = r.json()
        pid = str(info.get("id") or "")
        created = time.time()
        gpu = ((info.get("gpu") or {}).get("displayName") or "availability") if isinstance(info, dict) else "availability"
        print(json.dumps({"event": "public_pod_created", "pod": pid, "cloud": cloud, "gpu": gpu, "cost": info.get("costPerHr"), "image": PUBLIC_IMAGE}), flush=True)

        runtime_seen = False
        runtime_deadline = time.time() + 900
        last = None
        while time.time() < runtime_deadline:
            state = base.api("GET", f"/pods/{pid}")
            if state.status_code == 200:
                data = state.json()
                runtime = data.get("runtime")
                runtime_seen = isinstance(runtime, dict)
                snap = {
                    "desired": data.get("desiredStatus"),
                    "runtime": runtime_seen,
                    "last": data.get("lastStatusChange"),
                    "gpu": ((data.get("gpu") or {}).get("displayName") if isinstance(data.get("gpu"), dict) else None),
                    "machine_id": ((data.get("machine") or {}).get("id") if isinstance(data.get("machine"), dict) else None),
                }
                if snap != last:
                    print("public_pod_state=" + json.dumps({"pod": pid, "cloud": cloud, **snap}, ensure_ascii=False), flush=True)
                    last = snap
                if runtime_seen:
                    break
            time.sleep(15)
        if not runtime_seen:
            attempts.append({"cloud": cloud, "pod": pid, "result": "no_runtime_public_image"})
            base.delete_pod(pid)
            continue

        worker_deadline = time.time() + 1500
        chosen = None
        while time.time() < worker_deadline:
            now = datetime.now(timezone.utc)
            for worker in Worker.all(connection=base.redis):
                hb = worker.last_heartbeat
                if not hb or worker.name in before:
                    continue
                if hb.tzinfo is None:
                    hb = hb.replace(tzinfo=timezone.utc)
                if hb.timestamp() >= created - 5 and (now - hb).total_seconds() <= 60 and worker.get_state() in {"idle", "busy"}:
                    chosen = worker.name
                    break
            if chosen:
                break
            time.sleep(15)
        if chosen:
            actual = base.api("GET", f"/pods/{pid}").json()
            actual_gpu = ((actual.get("gpu") or {}).get("displayName") if isinstance(actual.get("gpu"), dict) else gpu) or gpu
            attempts.append({"cloud": cloud, "pod": pid, "gpu": actual_gpu, "result": "fresh_worker", "worker": chosen})
            print(json.dumps({"event": "public_worker_live", "pod": pid, "cloud": cloud, "gpu": actual_gpu, "worker": chosen}), flush=True)
            return pid, cloud, actual_gpu, chosen, attempts
        attempts.append({"cloud": cloud, "pod": pid, "result": "runtime_without_rq_after_bootstrap"})
        base.delete_pod(pid)
    raise RuntimeError("Public-image bootstrap could not produce a fresh RQ worker")


def main() -> None:
    base.clean_stale()
    base_env, mount = base.safe_template_env()
    bundle_url, bucket, key = upload_source_bundle(base_env)
    chosen_pid = None
    attempts = []
    status = {"source_sha": base.SOURCE_SHA, "run_id": base.RUN_ID, "status": "starting", "worker_image": PUBLIC_IMAGE}
    try:
        pid, cloud, gpu, worker, attempts = find_public_worker(base_env, mount, bundle_url)
        chosen_pid = pid
        Path("pod-id.txt").write_text(pid)
        status.update({"status": "worker_live", "pod_id": pid, "cloud": cloud, "gpu": gpu, "worker": worker})
        json.dump(status, open("round16-status.json", "w"), indent=2)
        result = base.enqueue_and_wait(cloud, gpu)
        json.dump(result, open("focused-result.json", "w"), ensure_ascii=False, indent=2)
        report = base.download_artifacts(result, base_env)
        base.validate_gold(report)
        status.update({"status": "gold_pass", "completed_count": report.get("completed_count")})
        json.dump(status, open("round16-status.json", "w"), indent=2)
        print("ROUND16_GOLD_PASS=true", flush=True)
    finally:
        json.dump(attempts, open("capacity-attempts.json", "w"), ensure_ascii=False, indent=2)
        if chosen_pid:
            base.delete_pod(chosen_pid)
        try:
            s3 = boto3.client(
                "s3",
                region_name=base_env.get("AWS_REGION") or "us-east-1",
                aws_access_key_id=base_env["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=base_env["AWS_SECRET_ACCESS_KEY"],
            )
            s3.delete_object(Bucket=bucket, Key=key)
            print("source_bundle_deleted=true", flush=True)
        except Exception as exc:
            print(f"source_bundle_delete_error={type(exc).__name__}", flush=True)


if __name__ == "__main__":
    main()
