from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rq import Worker

import round14_capacity_failover as base


# Compatibility-first ordering. Lower-cost creator-workload GPUs come first;
# larger datacenter GPUs are fallbacks, not the default.
GPU_PRIORITY = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L4",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40",
    "NVIDIA L40S",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
]
GPU_RANK = {gpu: i for i, gpu in enumerate(GPU_PRIORITY)}
STOCK_RANK = {"High": 0, "Medium": 1, "Low": 2, "": 3, None: 4}


def _template_registry_auth() -> str | None:
    r = base.api("GET", "/templates")
    r.raise_for_status()
    template = next((x for x in r.json() if x.get("name") == base.TEMPLATE_NAME), None)
    if not isinstance(template, dict):
        return None
    value = template.get("containerRegistryAuthId")
    print(f"template_registry_auth_present={bool(value)}", flush=True)
    return str(value) if value else None


def _load_live_candidates() -> list[tuple[str, str, str, str]]:
    path = Path(os.environ.get("RUNPOD_DATACENTER_STOCK", "datacenter-stock.json"))
    if not path.exists():
        raise RuntimeError(f"Missing live RunPod datacenter stock: {path}")
    raw = json.loads(path.read_text())
    candidates: list[tuple[str, str, str, str]] = []
    for dc in raw if isinstance(raw, list) else []:
        dc_id = str(dc.get("id") or "")
        if not dc_id:
            continue
        for item in dc.get("gpuAvailability") or []:
            gpu = str(item.get("gpuId") or "")
            stock = item.get("stockStatus")
            if gpu not in GPU_RANK:
                continue
            # Empty stock means RunPod lists the GPU but does not claim live stock.
            # Keep it only as a last-resort candidate after High/Medium/Low.
            candidates.append((dc_id, gpu, str(stock or ""), str(dc.get("location") or "")))

    # Deduplicate and sort by stock first, then our cost/compatibility preference.
    uniq = {(dc, gpu): (dc, gpu, stock, loc) for dc, gpu, stock, loc in candidates}
    ordered = list(uniq.values())
    ordered.sort(key=lambda x: (STOCK_RANK.get(x[2], 4), GPU_RANK.get(x[1], 999), x[0]))

    max_candidates = int(os.environ.get("ROUND20_MAX_CANDIDATES", "24"))
    ordered = ordered[:max_candidates]
    print("round20_candidates=" + json.dumps([
        {"dc": dc, "gpu": gpu, "stock": stock, "location": loc}
        for dc, gpu, stock, loc in ordered
    ], ensure_ascii=False), flush=True)
    return ordered


def find_worker_dynamic(base_env: dict, mount: str):
    before = {w.name for w in Worker.all(connection=base.redis)}
    env = base.build_env(base_env)
    registry_auth = _template_registry_auth()
    candidates = _load_live_candidates()
    if not candidates:
        raise RuntimeError("Live RunPod stock returned no compatible GPU candidates")

    attempts: list[dict] = []
    runtime_timeout = int(os.environ.get("ROUND20_RUNTIME_TIMEOUT_SEC", "90"))
    rq_timeout = int(os.environ.get("ROUND20_RQ_TIMEOUT_SEC", "480"))

    # Secure first for each live candidate, then Community. Never more than one Pod alive.
    for dc, gpu, stock, location in candidates:
        for cloud in ("SECURE", "COMMUNITY"):
            payload = {
                "name": base.POD_PREFIX,
                "imageName": base.IMMUTABLE_IMAGE,
                "cloudType": cloud,
                "computeType": "GPU",
                "gpuCount": 1,
                "gpuTypeIds": [gpu],
                "gpuTypePriority": "custom",
                "dataCenterIds": [dc],
                "dataCenterPriority": "custom",
                "interruptible": False,
                "containerDiskInGb": 30,
                "volumeInGb": 20,
                "volumeMountPath": mount,
                "env": env,
            }
            if registry_auth:
                payload["containerRegistryAuthId"] = registry_auth

            r = base.api("POST", "/pods", json=payload)
            if r.status_code != 201:
                attempts.append({
                    "cloud": cloud, "datacenter": dc, "gpu": gpu, "stock": stock,
                    "result": "create_rejected", "http": r.status_code,
                })
                print(f"round20_create_rejected cloud={cloud} dc={dc} gpu={gpu} stock={stock} http={r.status_code}", flush=True)
                continue

            info = r.json()
            pid = str(info.get("id") or "")
            created = time.time()
            print("round20_pod_created=" + json.dumps({
                "pod": pid, "cloud": cloud, "datacenter": dc, "location": location,
                "gpu": gpu, "stock": stock, "cost": info.get("costPerHr"),
                "image": info.get("imageName"),
            }, ensure_ascii=False), flush=True)

            runtime_seen = False
            machine_id = None
            deadline = time.time() + runtime_timeout
            last_snap = None
            while time.time() < deadline:
                state = base.api("GET", f"/pods/{pid}?includeMachine=true")
                if state.status_code == 200:
                    data = state.json()
                    runtime = data.get("runtime")
                    runtime_seen = isinstance(runtime, dict)
                    machine = data.get("machine") if isinstance(data.get("machine"), dict) else {}
                    machine_id = data.get("machineId") or machine.get("id") or machine_id
                    snap = {
                        "pod": pid, "cloud": cloud, "dc": dc, "gpu": gpu,
                        "desired": data.get("desiredStatus"), "runtime": runtime_seen,
                        "machine_id": machine_id, "last": data.get("lastStatusChange"),
                    }
                    if snap != last_snap:
                        print("round20_state=" + json.dumps(snap, ensure_ascii=False), flush=True)
                        last_snap = snap
                    if runtime_seen:
                        break
                else:
                    print(f"round20_pod_get_http={state.status_code} pod={pid}", flush=True)
                time.sleep(10)

            if not runtime_seen:
                attempts.append({
                    "cloud": cloud, "datacenter": dc, "gpu": gpu, "stock": stock,
                    "pod": pid, "machine_id": machine_id,
                    "result": f"no_runtime_{runtime_timeout}s",
                })
                base.delete_pod(pid)
                time.sleep(2)
                continue

            chosen_worker = None
            worker_deadline = time.time() + rq_timeout
            while time.time() < worker_deadline:
                now = datetime.now(timezone.utc)
                for worker in Worker.all(connection=base.redis):
                    hb = worker.last_heartbeat
                    if not hb or worker.name in before:
                        continue
                    if hb.tzinfo is None:
                        hb = hb.replace(tzinfo=timezone.utc)
                    age = (now - hb).total_seconds()
                    if hb.timestamp() >= created - 5 and age <= 45 and worker.get_state() in {"idle", "busy"}:
                        chosen_worker = worker.name
                        break
                if chosen_worker:
                    break
                time.sleep(10)

            if chosen_worker:
                attempts.append({
                    "cloud": cloud, "datacenter": dc, "gpu": gpu, "stock": stock,
                    "pod": pid, "machine_id": machine_id,
                    "result": "fresh_worker", "worker": chosen_worker,
                })
                Path("capacity-attempts.json").write_text(json.dumps(attempts, indent=2))
                print("round20_chosen=" + json.dumps({
                    "pod": pid, "cloud": cloud, "datacenter": dc, "gpu": gpu,
                    "stock": stock, "worker": chosen_worker,
                }, ensure_ascii=False), flush=True)
                return pid, cloud, f"{gpu}@{dc}", chosen_worker, attempts

            attempts.append({
                "cloud": cloud, "datacenter": dc, "gpu": gpu, "stock": stock,
                "pod": pid, "machine_id": machine_id,
                "result": f"runtime_without_rq_{rq_timeout}s",
            })
            base.delete_pod(pid)
            time.sleep(2)

    Path("capacity-attempts.json").write_text(json.dumps(attempts, indent=2))
    raise RuntimeError("Round20 dynamic RunPod allocation could not produce a fresh worker")


base.find_worker = find_worker_dynamic
base.main()
