from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from rq import Worker

import round14_capacity_failover as base


def find_worker_datacenter_pinned(base_env: dict, mount: str):
    """Acquire exactly one healthy worker at a time using live-stock datacenter pins."""
    before = {w.name for w in Worker.all(connection=base.redis)}
    env = base.build_env(base_env)

    # Live stock probe on 2026-08-23 reported EU-RO-1 RTX 4090 as High.
    # H100 AP-IN-1 is a compatibility-safe fallback and was also High.
    candidates = [
        ("SECURE", "EU-RO-1", "NVIDIA GeForce RTX 4090"),
        ("COMMUNITY", "EU-RO-1", "NVIDIA GeForce RTX 4090"),
        ("SECURE", "AP-IN-1", "NVIDIA H100 80GB HBM3"),
        ("COMMUNITY", "AP-IN-1", "NVIDIA H100 80GB HBM3"),
    ]

    attempts: list[dict] = []
    for cloud, dc, gpu in candidates:
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

        r = base.api("POST", "/pods", json=payload)
        if r.status_code != 201:
            attempts.append({
                "cloud": cloud,
                "datacenter": dc,
                "gpu": gpu,
                "result": "create_rejected",
                "http": r.status_code,
            })
            print(
                f"create_rejected cloud={cloud} dc={dc} gpu={gpu} http={r.status_code}",
                flush=True,
            )
            continue

        info = r.json()
        pid = str(info.get("id") or "")
        created = time.time()
        print(json.dumps({
            "event": "dc_pinned_pod_created",
            "pod": pid,
            "cloud": cloud,
            "datacenter": dc,
            "gpu": gpu,
            "cost": info.get("costPerHr"),
            "image": info.get("imageName"),
        }), flush=True)

        runtime_seen = False
        machine_id = None
        runtime_deadline = time.time() + 300
        last_snap = None
        while time.time() < runtime_deadline:
            state = base.api("GET", f"/pods/{pid}?includeMachine=true")
            if state.status_code == 200:
                data = state.json()
                runtime_seen = isinstance(data.get("runtime"), dict)
                machine = data.get("machine") if isinstance(data.get("machine"), dict) else {}
                machine_id = data.get("machineId") or machine.get("id") or machine_id
                snap = {
                    "pod": pid,
                    "cloud": cloud,
                    "datacenter": dc,
                    "gpu": gpu,
                    "desired": data.get("desiredStatus"),
                    "runtime": runtime_seen,
                    "machine_id": machine_id,
                    "last": data.get("lastStatusChange"),
                }
                if snap != last_snap:
                    print("dc_pinned_state=" + json.dumps(snap, ensure_ascii=False), flush=True)
                    last_snap = snap
                if runtime_seen:
                    break
            else:
                print(f"pod_get_http={state.status_code} pod={pid}", flush=True)
            time.sleep(10)

        if not runtime_seen:
            attempts.append({
                "cloud": cloud,
                "datacenter": dc,
                "gpu": gpu,
                "pod": pid,
                "machine_id": machine_id,
                "result": "no_runtime_300s",
            })
            base.delete_pod(pid)
            continue

        chosen_worker = None
        worker_deadline = time.time() + 480
        while time.time() < worker_deadline:
            now = datetime.now(timezone.utc)
            for worker in Worker.all(connection=base.redis):
                hb = worker.last_heartbeat
                if not hb or worker.name in before:
                    continue
                if hb.tzinfo is None:
                    hb = hb.replace(tzinfo=timezone.utc)
                if (
                    hb.timestamp() >= created - 5
                    and (now - hb).total_seconds() <= 45
                    and worker.get_state() in {"idle", "busy"}
                ):
                    chosen_worker = worker.name
                    break
            if chosen_worker:
                break
            time.sleep(10)

        if chosen_worker:
            attempts.append({
                "cloud": cloud,
                "datacenter": dc,
                "gpu": gpu,
                "pod": pid,
                "machine_id": machine_id,
                "result": "fresh_worker",
                "worker": chosen_worker,
            })
            print("dc_pinned_chosen=" + json.dumps({
                "pod": pid,
                "cloud": cloud,
                "datacenter": dc,
                "gpu": gpu,
                "machine_id": machine_id,
                "worker": chosen_worker,
            }), flush=True)
            return pid, cloud, f"{gpu}@{dc}", chosen_worker, attempts

        attempts.append({
            "cloud": cloud,
            "datacenter": dc,
            "gpu": gpu,
            "pod": pid,
            "machine_id": machine_id,
            "result": "runtime_without_rq_480s",
        })
        base.delete_pod(pid)

    raise RuntimeError("Datacenter-pinned RunPod allocation could not produce a fresh worker")


base.find_worker = find_worker_datacenter_pinned
base.main()
