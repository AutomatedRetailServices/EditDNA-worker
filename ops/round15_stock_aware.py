from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from rq import Worker

import round14_capacity_failover as base


def _template_registry_auth() -> str | None:
    r = base.api("GET", "/templates")
    r.raise_for_status()
    template = next((x for x in r.json() if x.get("name") == base.TEMPLATE_NAME), None)
    if not isinstance(template, dict):
        return None
    value = template.get("containerRegistryAuthId")
    if value:
        print("template_registry_auth_present=true", flush=True)
        return str(value)
    print("template_registry_auth_present=false", flush=True)
    return None


def find_worker_stock_aware(base_env: dict, mount: str):
    before = {w.name for w in Worker.all(connection=base.redis)}
    registry_auth = _template_registry_auth()
    env = base.build_env(base_env)
    gpu_ids = [
        "NVIDIA GeForce RTX 5090",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA L40S",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA L40",
        "NVIDIA H100 PCIe",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA A40",
        "NVIDIA RTX A6000",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA L4",
        "NVIDIA RTX 5000 Ada Generation",
        "NVIDIA GeForce RTX 4080 SUPER",
    ]
    attempts: list[dict] = []
    for cloud in ("SECURE", "COMMUNITY"):
        payload = {
            "name": base.POD_PREFIX,
            "imageName": base.IMMUTABLE_IMAGE,
            "cloudType": cloud,
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": gpu_ids,
            "gpuTypePriority": "availability",
            "dataCenterPriority": "availability",
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
            print(f"availability_create_rejected cloud={cloud} http={r.status_code}", flush=True)
            attempts.append({"cloud": cloud, "result": "create_rejected", "http": r.status_code})
            continue
        info = r.json()
        pid = str(info.get("id") or "")
        created = time.time()
        print(json.dumps({"event": "availability_pod_created", "pod": pid, "cloud": cloud, "cost": info.get("costPerHr"), "image": info.get("imageName")}), flush=True)

        runtime_seen = False
        selected_gpu = None
        runtime_deadline = time.time() + 600
        last_snap = None
        while time.time() < runtime_deadline:
            state = base.api("GET", f"/pods/{pid}?includeMachine=true")
            if state.status_code == 200:
                data = state.json()
                runtime_seen = isinstance(data.get("runtime"), dict)
                machine = data.get("machine") if isinstance(data.get("machine"), dict) else {}
                selected_gpu = (
                    data.get("gpuTypeId")
                    or machine.get("gpuTypeId")
                    or machine.get("gpuDisplayName")
                    or selected_gpu
                )
                snap = {
                    "desired": data.get("desiredStatus"),
                    "runtime": runtime_seen,
                    "last": data.get("lastStatusChange"),
                    "gpu": selected_gpu,
                    "machine_id": machine.get("id"),
                }
                if snap != last_snap:
                    print("availability_pod_state=" + json.dumps({"pod": pid, "cloud": cloud, **snap}, ensure_ascii=False), flush=True)
                    last_snap = snap
                if runtime_seen:
                    break
            else:
                print(f"pod_get_http={state.status_code}", flush=True)
            time.sleep(15)
        if not runtime_seen:
            attempts.append({"cloud": cloud, "pod": pid, "gpu": selected_gpu, "result": "no_runtime_600s"})
            base.delete_pod(pid)
            continue

        chosen_worker = None
        worker_deadline = time.time() + 600
        while time.time() < worker_deadline:
            now = datetime.now(timezone.utc)
            for worker in Worker.all(connection=base.redis):
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
            gpu_label = str(selected_gpu or "availability-selected")
            attempts.append({"cloud": cloud, "pod": pid, "gpu": gpu_label, "result": "fresh_worker", "worker": chosen_worker})
            print("availability_chosen=" + json.dumps({"pod": pid, "cloud": cloud, "gpu": gpu_label, "worker": chosen_worker}), flush=True)
            return pid, cloud, gpu_label, chosen_worker, attempts
        attempts.append({"cloud": cloud, "pod": pid, "gpu": selected_gpu, "result": "runtime_without_rq_600s"})
        base.delete_pod(pid)

    raise RuntimeError("Stock-aware RunPod allocation could not produce a fresh worker")


base.find_worker = find_worker_stock_aware
base.main()
