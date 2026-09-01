"""D-041 follow-up: read-only inspection of RunPod endpoint `xxu7autt8mv2rn`'s
live GPU/placement configuration. Infrastructure/orchestration only -- no
CutSell editorial code touched, no GPU job submitted, no Video00 run,
zero cost beyond a handful of GET requests.

Why this exists: RAW 33453836301 and the D-041 hardened retry (33457835750)
both proved the orchestration layer works correctly (readiness detection,
IN_QUEUE stall detection, bounded retry, teardown, classification) but hit a
persistent RunPod worker-placement/capacity failure -- two independent
health jobs, on two independently fresh endpoint rolls, each sat IN_QUEUE
for a full 5 minutes with no worker ever assigned. Before proposing any GPU
fallback policy, the actual live endpoint/template configuration needs to be
read, not guessed -- the repository itself pins no GPU type anywhere (the
endpoint PATCH payload in the workflow never sets `gpuIds`), so this is the
only way to answer "what GPU class is this endpoint actually pinned to."

Reuses `runpod_orchestration.UrllibTransport` (stdlib `urllib.request` only,
no new dependency) rather than duplicating HTTP plumbing.

Safety: only an explicit ALLOWLIST of known-non-secret fields is ever
printed (never a denylist/mask-and-hope) -- an endpoint response has no
secrets to begin with, but a *template* response's `env` dict does (AWS
keys, GEMINI_API_KEY), so `env` is dropped unconditionally before anything
from a template response is printed, and only allowlisted keys are echoed.
"""
from __future__ import annotations

import json
import os
import sys

from runpod_orchestration import RUNPOD_REST_BASE, UrllibTransport

# Only fields with no possible secret content. Deliberately an allowlist,
# not a denylist -- an unrecognized field is dropped rather than guessed at.
_SAFE_ENDPOINT_KEYS = {
    "id",
    "name",
    "templateId",
    "workersMin",
    "workersMax",
    "gpuIds",
    "scalerType",
    "scalerValue",
    "idleTimeout",
    "executionTimeoutMs",
    "networkVolumeId",
    "locations",
    "dataCenterIds",
    "computeType",
    "allowedCudaVersions",
}
_SAFE_TEMPLATE_KEYS = {
    "id",
    "name",
    "imageName",
    "containerDiskInGb",
    "isServerless",
    "isPublic",
    "category",
    "volumeInGb",
    "volumeMountPath",
}


def filter_safe(obj: dict | None, allowlist: set[str]) -> dict:
    """Allowlist filter used for both endpoint and template responses --
    unconditionally drops everything not explicitly named safe, `env`
    (secrets) included, whether or not `env` is in the allowlist."""
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items() if k in allowlist}


def _print_safe(label: str, obj: dict | None, allowlist: set[str]) -> None:
    if not obj:
        print(f"--- {label}: absent/unavailable ---")
        return
    # Key NAMES carry no secret content (unlike values), so print the full
    # set of top-level keys the live response actually has -- this is how a
    # field this allowlist didn't anticipate (e.g. the real GPU-selection
    # field name, whatever RunPod actually calls it) gets discovered rather
    # than silently dropped and never noticed.
    print(f"--- {label}: all top-level key names present (values never shown here) ---")
    print(sorted(obj.keys()))
    print(f"--- {label} (allowlisted fields only; secrets/env never printed) ---")
    print(json.dumps(filter_safe(obj, allowlist), indent=2, default=str))


def main() -> int:
    endpoint_id = os.environ.get("ENDPOINT_ID", "xxu7autt8mv2rn")
    api_key = os.environ["RUNPOD_API_KEY"]
    transport = UrllibTransport()
    headers = {"Authorization": f"Bearer {api_key}"}

    resp = transport.request("GET", f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}", headers=headers)
    print(f"GET /endpoints/{endpoint_id} -> {resp.status_code}")
    endpoint_obj = resp.json_body if resp.status_code == 200 and isinstance(resp.json_body, dict) else None

    list_resp = transport.request("GET", f"{RUNPOD_REST_BASE}/endpoints", headers=headers)
    print(f"GET /endpoints (list) -> {list_resp.status_code}")
    matched_from_list = None
    if list_resp.status_code == 200 and isinstance(list_resp.json_body, list):
        matched_from_list = next(
            (e for e in list_resp.json_body if isinstance(e, dict) and e.get("id") == endpoint_id), None
        )

    _print_safe("endpoint (direct GET)", endpoint_obj, _SAFE_ENDPOINT_KEYS)
    _print_safe("endpoint (from list)", matched_from_list, _SAFE_ENDPOINT_KEYS)

    template_id = (endpoint_obj or {}).get("templateId") or (matched_from_list or {}).get("templateId")
    if template_id:
        t_resp = transport.request("GET", f"{RUNPOD_REST_BASE}/templates/{template_id}", headers=headers)
        print(f"GET /templates/{template_id} -> {t_resp.status_code}")
        template_obj = t_resp.json_body if t_resp.status_code == 200 and isinstance(t_resp.json_body, dict) else None
        _print_safe("current template", template_obj, _SAFE_TEMPLATE_KEYS)
    else:
        print("--- current template: no templateId found on endpoint response ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
