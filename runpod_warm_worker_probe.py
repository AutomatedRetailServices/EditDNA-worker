"""RunPod Support follow-up: warm-worker (workersMin=1) health-only probe.

Distinct from runpod_fast_isolation_probe.py's cold/on-demand test
(workersMin=0): this forces RunPod to provision one warm worker
independently of any job arrival, to test whether cold/on-demand
provisioning specifically is the bottleneck, or whether the account/
provider cannot provision a compatible worker at all regardless.

A prior version of this probe tried to directly observe worker
provisioning via GET /v1/endpoints/{id}/workers before ever submitting a
job -- that endpoint returned HTTP 400 on this account (does not exist at
that path). Per explicit instruction, this version instead uses the
health job's OWN lifecycle (IN_QUEUE -> IN_PROGRESS -> COMPLETED/FAILED)
as the worker-provisioning signal: submit exactly ONE health job right
after workersMin/workersMax=1 is confirmed via a read-after-write GET,
and record every status transition and its timing.

Infrastructure/orchestration only, no CutSell editorial code touched.
Never submits Video00 under any code path. No second health retry after
this test (submit_and_poll_health is called once, directly -- never
through run_with_bounded_retry).

Creates a new template + new endpoint, submits ONE health job, collects
full evidence, and ALWAYS tears both down afterward -- regardless of
outcome. Reuses runpod_orchestration.py's and runpod_fast_isolation_
probe.py's already-tested primitives rather than re-implementing them.
"""
from __future__ import annotations

import json
import os
import sys
import time

from runpod_fast_isolation_probe import (
    create_temp_template,
    resolve_current_gpu_type_ids,
    teardown,
)
from runpod_orchestration import (
    RUNPOD_REST_BASE,
    RUNPOD_SERVERLESS_BASE,
    OrchestrationEvent,
    UrllibTransport,
    _default_log,
    submit_and_poll_health,
    wait_for_endpoint_ready,
)


def create_warm_temp_endpoint(
    transport, api_key: str, *, template_id: str, gpu_type_ids: list[str], run_id: str,
    workers_min: int, workers_max: int, log,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "templateId": template_id,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpu_type_ids,
        "workersMin": workers_min,
        "workersMax": workers_max,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 2,
        "idleTimeout": 5,
        "executionTimeoutMs": 900000,
        "flashboot": True,
        # No region/datacenter restriction, per instruction.
        "name": f"cutsell-warm-worker-probe-{run_id}",
    }
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/endpoints", headers=headers, json_body=payload)
    if resp.status_code not in (200, 201) or not (resp.json_body or {}).get("id"):
        raise RuntimeError(f"endpoint creation failed: http {resp.status_code} body {resp.json_body}")
    endpoint_id = resp.json_body["id"]
    log(OrchestrationEvent(
        "warm_temp_endpoint_created", time.time(),
        {"endpoint_id": endpoint_id, "gpu_type_ids": gpu_type_ids, "workers_min": workers_min, "workers_max": workers_max},
    ))
    return endpoint_id


def fetch_job_payload(transport, endpoint_id: str, api_key: str, job_id: str) -> dict | None:
    """One extra, harmless, read-only GET after polling has already
    finished, purely to retrieve the job's full final body (health output:
    device_name/compute_capability/incompatibility_reason/etc, or the
    terminal-failure body) for reporting -- submit_and_poll_health's own
    return value does not carry the raw payload."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = transport.request(
        "GET", f"{RUNPOD_SERVERLESS_BASE}/{endpoint_id}/status/{job_id}", headers=headers,
    )
    return resp.json_body if resp.status_code == 200 else None


def main() -> int:
    api_key = os.environ["RUNPOD_API_KEY"]
    image = os.environ["PROBE_IMAGE"]
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    transport = UrllibTransport()

    transitions: list[dict] = []

    def log(event: OrchestrationEvent) -> None:
        _default_log(event)
        if event.name == "health_status":
            transitions.append({"at": event.at, **event.fields})

    evidence: dict = {"probe_started_at": time.time(), "image": image, "workers_min": 1, "workers_max": 1}
    template_id = None
    endpoint_id = None
    job_id = None

    try:
        evidence["resolved_gpu_type_ids"] = resolve_current_gpu_type_ids(transport, api_key)

        template_id = create_temp_template(transport, api_key, image=image, run_id=run_id)
        evidence["template_id"] = template_id

        endpoint_id = create_warm_temp_endpoint(
            transport, api_key, template_id=template_id,
            gpu_type_ids=evidence["resolved_gpu_type_ids"], run_id=run_id,
            workers_min=1, workers_max=1, log=log,
        )
        evidence["endpoint_id"] = endpoint_id
        evidence["endpoint_created_at"] = time.time()

        # Confirms workersMin=1/workersMax=1 via a direct read-after-write
        # GET, same rigor as every prior probe in this investigation.
        readiness = wait_for_endpoint_ready(
            transport, endpoint_id, api_key, template_id, 1, log=log,
        )
        evidence["endpoint_ready"] = readiness.ready
        evidence["endpoint_ready_at"] = time.time()
        evidence["workers_min_max_confirmed"] = readiness.ready
        evidence["endpoint_readiness_classification"] = readiness.classification

        if not readiness.ready:
            evidence["health"] = None
            evidence["health_payload"] = None
            evidence["decision"] = "B"  # never even reached a stable, confirmed endpoint config
        else:
            health_submitted_at = time.time()
            # Exactly ONE health job, called directly -- no retry wrapper,
            # no second attempt after this, per instruction.
            health = submit_and_poll_health(
                transport, endpoint_id, api_key, is_retry_attempt=False, log=log,
            )
            job_id = health.job_id
            health_result_at = time.time()

            time_to_in_progress_s = next(
                (t["elapsed_s"] for t in transitions if t.get("status") == "IN_PROGRESS"), None,
            )
            worker_id_seen = next(
                (t["worker_id"] for t in transitions if t.get("worker_id")), None,
            )
            payload = fetch_job_payload(transport, endpoint_id, api_key, job_id) if job_id else None

            evidence["health_job_id"] = job_id
            evidence["health_submitted_at"] = health_submitted_at
            evidence["status_transitions"] = transitions
            evidence["time_in_queue_s"] = health.time_in_queue_s
            evidence["time_to_in_progress_s"] = time_to_in_progress_s
            evidence["worker_id"] = worker_id_seen
            evidence["health"] = {
                "classification": health.classification,
                "passed": health.passed,
                "final_status": health.final_status,
                "detail": health.detail,
            }
            evidence["health_payload"] = payload
            evidence["health_result_at"] = health_result_at

            if health.passed:
                evidence["decision"] = "A"
            elif health.classification in ("WORKER_PROVISIONING_STALLED", "CAPACITY_UNAVAILABLE"):
                evidence["decision"] = "B"
            elif health.classification == "HEALTH_APP_FAILURE":
                # Distinguish C (GPU outside allowlist) vs D (allowed GPU,
                # CUDA/health still fails) using the enhanced diagnostics
                # payload when present -- fails safely to "D" (report exact
                # evidence) if the payload can't answer this.
                device_name = (payload or {}).get("output", {}).get("device_name") if isinstance(payload, dict) else None
                gpu_type_ids = evidence["resolved_gpu_type_ids"]
                if device_name and device_name not in gpu_type_ids:
                    evidence["decision"] = "C"
                else:
                    evidence["decision"] = "D"
            else:
                evidence["decision"] = "unclassified"

    finally:
        evidence["teardown"] = teardown(
            transport, api_key, endpoint_id=endpoint_id, template_id=template_id, job_id=job_id,
        )
        evidence["probe_finished_at"] = time.time()

    print("=== WARM WORKER HEALTH-ONLY PROBE EVIDENCE (JSON) ===")
    print(json.dumps(evidence, indent=2, default=str))

    # Never submits Video00 -- this script has no code path that ever calls
    # anything but the "health" op. Exit 0 unconditionally: this is a
    # diagnostic probe, not a gate -- the caller reads the evidence JSON.
    return 0


if __name__ == "__main__":
    sys.exit(main())
