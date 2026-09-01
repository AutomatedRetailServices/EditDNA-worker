"""RunPod Support follow-up: warm-worker (workersMin=1) health-only probe.

Distinct from runpod_fast_isolation_probe.py's cold/on-demand test
(workersMin=0): this forces RunPod to provision one warm worker
independently of any job arrival, to test whether cold/on-demand
provisioning specifically is the bottleneck, or whether the account/
provider cannot provision a compatible worker at all regardless.

Infrastructure/orchestration only, no CutSell editorial code touched.
Never submits Video00; only ever calls the cheap {"op":"health"} job, and
only AFTER a worker is directly confirmed provisioned via the endpoint's
own workers-list API -- per the explicit instruction this follows, health
is never submitted on a mere assumption that a worker exists.

Creates a new template + new endpoint (workersMin=1, workersMax=1,
FlashBoot enabled, same compatible-only gpuTypeIds allowlist, same current
production image), waits for a directly-observed worker for a bounded
period, submits ONE health job only if one was observed, collects full
evidence, and ALWAYS tears both down afterward -- regardless of outcome.

Reuses runpod_orchestration.py's and runpod_fast_isolation_probe.py's
already-tested primitives rather than re-implementing them.
"""
from __future__ import annotations

import json
import os
import sys
import time

from runpod_fast_isolation_probe import (
    COMPATIBLE_GPU_TYPE_IDS,
    create_temp_template,
    resolve_current_gpu_type_ids,
    teardown,
)
from runpod_orchestration import (
    RUNPOD_REST_BASE,
    OrchestrationEvent,
    UrllibTransport,
    _default_log,
    submit_and_poll_health,
    wait_for_endpoint_ready,
)


def _log(event: OrchestrationEvent) -> None:
    _default_log(event)


def create_warm_temp_endpoint(
    transport, api_key: str, *, template_id: str, gpu_type_ids: list[str], run_id: str,
    workers_min: int, workers_max: int,
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
        "name": f"cutsell-warm-worker-probe-{run_id}",
    }
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/endpoints", headers=headers, json_body=payload)
    if resp.status_code not in (200, 201) or not (resp.json_body or {}).get("id"):
        raise RuntimeError(f"endpoint creation failed: http {resp.status_code} body {resp.json_body}")
    endpoint_id = resp.json_body["id"]
    _log(OrchestrationEvent(
        "warm_temp_endpoint_created", time.time(),
        {"endpoint_id": endpoint_id, "gpu_type_ids": gpu_type_ids, "workers_min": workers_min, "workers_max": workers_max},
    ))
    return endpoint_id


def wait_for_worker_provisioned(
    transport, endpoint_id: str, api_key: str, *,
    timeout_s: float = 90.0, poll_interval_s: float = 5.0,
    now=time.time, sleep=time.sleep, log=_log,
) -> dict:
    """Directly observes (does not assume) whether RunPod has provisioned a
    worker for this endpoint, via GET /v1/endpoints/{id}/workers -- polled
    for a bounded period BEFORE any job is submitted. If that endpoint is
    unavailable on this account (non-200 on the very first probe), this is
    reported honestly as `observed_via: workers_endpoint_unavailable` rather
    than silently falling back to an assumption."""
    headers = {"Authorization": f"Bearer {api_key}"}
    start = now()
    workers_endpoint_available: bool | None = None
    last_status_code = None

    while True:
        elapsed = now() - start
        resp = transport.request("GET", f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}/workers", headers=headers)
        last_status_code = resp.status_code
        if workers_endpoint_available is None:
            workers_endpoint_available = resp.status_code == 200
            log(OrchestrationEvent(
                "workers_endpoint_probe", now(),
                {"status_code": resp.status_code, "available": workers_endpoint_available},
            ))
            if not workers_endpoint_available:
                return {
                    "observed_via": "workers_endpoint_unavailable",
                    "found": None,
                    "elapsed_s": elapsed,
                    "workers": None,
                    "http_status": last_status_code,
                }

        workers = resp.json_body if isinstance(resp.json_body, list) else []
        log(OrchestrationEvent("worker_poll", now(), {"elapsed_s": elapsed, "worker_count": len(workers)}))
        if workers:
            return {"observed_via": "workers_endpoint", "found": True, "elapsed_s": elapsed, "workers": workers}
        if elapsed >= timeout_s:
            return {"observed_via": "workers_endpoint", "found": False, "elapsed_s": elapsed, "workers": []}
        sleep(poll_interval_s)


def main() -> int:
    api_key = os.environ["RUNPOD_API_KEY"]
    image = os.environ["PROBE_IMAGE"]
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    transport = UrllibTransport()

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
            workers_min=1, workers_max=1,
        )
        evidence["endpoint_id"] = endpoint_id
        evidence["endpoint_created_at"] = time.time()

        readiness = wait_for_endpoint_ready(
            transport, endpoint_id, api_key, template_id, 1, log=_log,
        )
        evidence["endpoint_ready"] = readiness.ready
        evidence["endpoint_ready_at"] = time.time()
        evidence["endpoint_readiness_classification"] = readiness.classification
        evidence["endpoint_readiness_elapsed_s"] = readiness.elapsed_s

        if not readiness.ready:
            evidence["worker_provisioning_precheck"] = None
            evidence["health"] = None
            evidence["decision_tree_case"] = "C"  # could not even reach a stable endpoint config
        else:
            precheck = wait_for_worker_provisioned(transport, endpoint_id, api_key, timeout_s=90.0, log=_log)
            evidence["worker_provisioning_precheck"] = precheck
            evidence["worker_precheck_result_at"] = time.time()

            if precheck.get("found") is True:
                # A worker was directly confirmed BEFORE submitting health --
                # exactly the gate the instruction requires.
                health = submit_and_poll_health(
                    transport, endpoint_id, api_key, is_retry_attempt=False, log=_log,
                )
                job_id = health.job_id
                evidence["health"] = {
                    "job_id": health.job_id,
                    "classification": health.classification,
                    "passed": health.passed,
                    "time_in_queue_s": health.time_in_queue_s,
                    "final_status": health.final_status,
                    "detail": health.detail,
                }
                evidence["decision_tree_case"] = "B" if health.passed else "E"
            elif precheck.get("found") is False:
                # workersMin=1 was API-confirmed, FlashBoot enabled, bounded
                # wait elapsed, and the workers-list endpoint (confirmed
                # available) never showed a worker. Per instructions, health
                # is NOT submitted on an unconfirmed worker.
                evidence["health"] = None
                evidence["decision_tree_case"] = "C"
            else:
                # workers-list endpoint itself unavailable on this account --
                # cannot confirm provisioning via direct API observation.
                # Per instructions ("ONLY after a worker is actually
                # provisioned"), health is not submitted without that
                # confirmation rather than assumed.
                evidence["health"] = None
                evidence["decision_tree_case"] = "unknown_no_workers_api"

        evidence["health_result_at"] = time.time()

    finally:
        evidence["teardown"] = teardown(
            transport, api_key, endpoint_id=endpoint_id, template_id=template_id, job_id=job_id,
        )
        evidence["probe_finished_at"] = time.time()

    print("=== WARM WORKER PROBE EVIDENCE (JSON) ===")
    print(json.dumps(evidence, indent=2, default=str))

    # Never submits Video00 -- this script has no code path that ever calls
    # anything but the "health" op. Exit 0 unconditionally: this is a
    # diagnostic probe, not a gate -- the caller reads the evidence JSON.
    return 0


if __name__ == "__main__":
    sys.exit(main())
