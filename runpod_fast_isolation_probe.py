"""Fast RunPod isolation probe -- fresh, temporary, health-only serverless
endpoint. Infrastructure/orchestration only, no CutSell editorial code
touched. Never submits Video00; only ever calls the cheap `{"op":"health"}`
job.

Purpose: determine whether the EXISTING production endpoint (xxu7autt8mv2rn)
is itself the problem (stale/unhealthy), or whether the underlying RunPod
account/provider cannot currently provision compatible capacity at all, or
is assigning GPU hardware outside our configured allowlist, by trying the
exact same recipe (current production image, explicit compatible-only
`gpuTypeIds`) on a brand-new endpoint that has no history of its own.

Creates a new template + new endpoint, submits ONE health job, collects full
evidence, and ALWAYS tears both down afterward -- regardless of outcome.
Reuses runpod_orchestration.py's already-tested Transport/polling/
classification primitives rather than re-implementing them.
"""
from __future__ import annotations

import json
import os
import sys
import time

from runpod_orchestration import (
    RUNPOD_REST_BASE,
    RUNPOD_SERVERLESS_BASE,
    OrchestrationEvent,
    UrllibTransport,
    _default_log,
    cancel_job_if_active,
    submit_and_poll_health,
    wait_for_endpoint_ready,
)

# Explicit compatible-only allowlist (D-041 GPU-fallback-audit follow-up).
# Blackwell (sm_120) deliberately excluded -- the current production image
# (torch 2.6.0 / CUDA 12.4) does not support it.
COMPATIBLE_GPU_TYPE_IDS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
]


def _log(event: OrchestrationEvent) -> None:
    _default_log(event)


def resolve_current_gpu_type_ids(transport, api_key: str) -> list[str]:
    """Cross-check the compatible allowlist against RunPod's own live GPU
    catalog rather than assuming the literals are still current. Falls back
    to the known-good list (already confirmed live and in active use on the
    production endpoint moments before this probe) if the catalog call
    itself is unavailable -- never blocks the probe on this cross-check."""
    headers = {"Authorization": f"Bearer {api_key}"}
    for url in (f"{RUNPOD_REST_BASE}/gpuTypes", f"{RUNPOD_REST_BASE}/gpu-types"):
        resp = transport.request("GET", url, headers=headers)
        if resp.status_code == 200 and isinstance(resp.json_body, list):
            catalog_ids = {
                str(g.get("id") or "") for g in resp.json_body if isinstance(g, dict)
            }
            resolved = [g for g in COMPATIBLE_GPU_TYPE_IDS if g in catalog_ids]
            missing = [g for g in COMPATIBLE_GPU_TYPE_IDS if g not in catalog_ids]
            _log(OrchestrationEvent(
                "gpu_catalog_cross_check", time.time(),
                {"catalog_url": url, "resolved": resolved, "missing_from_catalog": missing},
            ))
            if resolved:
                return resolved
            break
    _log(OrchestrationEvent(
        "gpu_catalog_cross_check_unavailable_using_known_good_list", time.time(),
        {"list": COMPATIBLE_GPU_TYPE_IDS},
    ))
    return list(COMPATIBLE_GPU_TYPE_IDS)


def create_temp_template(transport, api_key: str, *, image: str, run_id: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    # Same masked storage credentials + env shape as the proven
    # cutsell-serverless-gpu-gate.yml recipe -- avoids re-discovering
    # import-time environment requirements through trial and error.
    payload = {
        "imageName": image,
        "name": f"cutsell-fast-isolation-{run_id}",
        "category": "NVIDIA",
        "containerDiskInGb": 40,
        "isPublic": False,
        "isServerless": True,
        "env": {
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_REGION": os.environ["AWS_REGION"],
            "S3_BUCKET": os.environ["S3_BUCKET"],
            "GEMINI_API_KEY": os.environ["GEMINI_API_KEY"],
            "CUTSELL_BRAIN_BACKEND": "runpod_local",
            "CUTSELL_EDITORIAL_MODE": "clean_cut",
            "CUTSELL_ASR_MODEL": "medium",
            "CUTSELL_CLEAN_CUT_JUDGE": "0",
            "CUTSELL_HYBRID_PROVIDER": "google",
            "CUTSELL_HYBRID_LLM_ENABLED": "1",
        },
        "ports": [],
        "volumeInGb": 0,
        "volumeMountPath": "/workspace",
    }
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/templates", headers=headers, json_body=payload)
    if resp.status_code not in (200, 201) or not (resp.json_body or {}).get("id"):
        raise RuntimeError(f"template creation failed: http {resp.status_code} body {resp.json_body}")
    template_id = resp.json_body["id"]
    _log(OrchestrationEvent("temp_template_created", time.time(), {"template_id": template_id}))
    return template_id


def create_temp_endpoint(transport, api_key: str, *, template_id: str, gpu_type_ids: list[str], run_id: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "templateId": template_id,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpu_type_ids,
        "workersMin": 0,
        "workersMax": 1,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 2,
        "idleTimeout": 5,
        "executionTimeoutMs": 900000,
        "flashboot": True,
        "name": f"cutsell-fast-isolation-{run_id}",
    }
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/endpoints", headers=headers, json_body=payload)
    if resp.status_code not in (200, 201) or not (resp.json_body or {}).get("id"):
        raise RuntimeError(f"endpoint creation failed: http {resp.status_code} body {resp.json_body}")
    endpoint_id = resp.json_body["id"]
    _log(OrchestrationEvent(
        "temp_endpoint_created", time.time(),
        {"endpoint_id": endpoint_id, "gpu_type_ids": gpu_type_ids},
    ))
    return endpoint_id


def teardown(transport, api_key: str, *, endpoint_id: str | None, template_id: str | None, job_id: str | None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    result = {"job_cancelled": False, "endpoint_deleted": False, "template_deleted": False}
    if endpoint_id and job_id:
        cancel_job_if_active(transport, endpoint_id, api_key, job_id, log=_log)
        result["job_cancelled"] = True
    if endpoint_id:
        resp = transport.request("DELETE", f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}", headers=headers)
        result["endpoint_deleted"] = resp.status_code in (200, 204)
        result["endpoint_delete_status"] = resp.status_code
    if template_id:
        resp = transport.request("DELETE", f"{RUNPOD_REST_BASE}/templates/{template_id}", headers=headers)
        result["template_deleted"] = resp.status_code in (200, 204)
        result["template_delete_status"] = resp.status_code
    _log(OrchestrationEvent("teardown_complete", time.time(), result))
    return result


def main() -> int:
    api_key = os.environ["RUNPOD_API_KEY"]
    image = os.environ["PROBE_IMAGE"]
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    transport = UrllibTransport()

    evidence: dict = {"probe_started_at": time.time(), "image": image}
    template_id = None
    endpoint_id = None
    job_id = None

    try:
        evidence["resolved_gpu_type_ids"] = resolve_current_gpu_type_ids(transport, api_key)

        template_id = create_temp_template(transport, api_key, image=image, run_id=run_id)
        evidence["template_id"] = template_id

        endpoint_id = create_temp_endpoint(
            transport, api_key, template_id=template_id,
            gpu_type_ids=evidence["resolved_gpu_type_ids"], run_id=run_id,
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
            evidence["health"] = None
            evidence["case"] = "B"  # never even reached a stable endpoint config
        else:
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

        evidence["health_result_at"] = time.time()

    finally:
        evidence["teardown"] = teardown(
            transport, api_key, endpoint_id=endpoint_id, template_id=template_id, job_id=job_id,
        )
        evidence["probe_finished_at"] = time.time()

    print("=== FAST ISOLATION PROBE EVIDENCE (JSON) ===")
    print(json.dumps(evidence, indent=2, default=str))

    # Never submits Video00 -- this script has no code path that ever calls
    # anything but the "health" op. Exit 0 unconditionally: this is a
    # diagnostic probe, not a gate -- the caller reads the evidence JSON.
    return 0


if __name__ == "__main__":
    sys.exit(main())
