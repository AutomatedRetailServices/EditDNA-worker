"""RunPod Pod template management (D-042 follow-up: CutSell-Pod-QA template,
cloned from the known-working EditDNA-Worker-2 template).

Read-only inspection first (`find_template_by_name`, `redact_template_env`),
write second (`create_pod_template`) -- never mutates the base template.
Reuses the same injectable `Transport` protocol `runpod_orchestration.py`
and `runpod_pod_provider.py` already established: fakes in tests, live
`UrllibTransport` only from a GitHub Actions step holding RUNPOD_API_KEY
(this sandbox has no direct RunPod network access).

Infrastructure/orchestration only -- no CutSell editorial logic here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from runpod_orchestration import RUNPOD_REST_BASE, LogFn, OrchestrationEvent, Transport, _default_log


def list_templates(transport: Transport, api_key: str) -> list[dict]:
    """Read-only GET /v1/templates. Returns [] (never raises) if the
    response isn't a parseable list -- callers treat that as "couldn't
    inspect", never as "no templates exist"."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = transport.request("GET", f"{RUNPOD_REST_BASE}/templates", headers=headers)
    if resp.status_code == 200 and isinstance(resp.json_body, list):
        return resp.json_body
    return []


def find_template_by_name(transport: Transport, api_key: str, name: str) -> Optional[dict]:
    """Exact-name match against the live template catalog. Returns the raw
    (unredacted) template dict, or None if not found -- never guesses."""
    for template in list_templates(transport, api_key):
        if isinstance(template, dict) and template.get("name") == name:
            return template
    return None


# Env var name fragments that mark a value as a name-only field on principle
# even if it doesn't look secret-shaped -- redaction here follows the same
# blanket policy runpod_endpoint_inspect.py already established for this
# repo: template env VALUES never reach stdout, regardless of whether a
# given one looks sensitive. Only the NAMES are ever shown.
def redact_template_env(template: dict) -> dict:
    """Returns a shallow copy of `template` with every `env` value replaced
    by a fixed redaction marker -- keys (names) preserved, sorted for a
    stable diff. Every other field is passed through as-is (image name,
    ports, disk/volume sizes, etc. are not secrets)."""
    out = dict(template)
    env = template.get("env") or {}
    out["env"] = {name: "<redacted>" for name in sorted(env.keys())}
    return out


def _normalize_ports(base_ports, required: tuple[str, ...]) -> list[str]:
    """RunPod's template `ports` field has appeared as either a list or a
    comma-separated string across API eras -- normalize whatever the base
    template has to a list, then ensure every port in `required` is
    present (order-preserving, no duplicates)."""
    if isinstance(base_ports, str):
        current = [p.strip() for p in base_ports.split(",") if p.strip()]
    elif isinstance(base_ports, list):
        current = [str(p) for p in base_ports]
    else:
        current = []
    for port in required:
        if port not in current:
            current.append(port)
    return current


@dataclass(frozen=True)
class PodTemplateOverrides:
    """Only the fields CutSell-Pod-QA is explicitly allowed to change from
    the base template. Everything else in the base is preserved verbatim
    -- see D-042's "prefer exact inheritance/parity over manually inventing
    configuration" directive."""

    name: str
    image: str
    start_command: Optional[list[str]] = None  # None => preserve base dockerStartCmd untouched
    required_ports: tuple[str, ...] = ("8080/http",)
    env_overrides: dict = field(default_factory=dict)  # merged over (not replacing) base env
    container_disk_gb: Optional[int] = None  # None => preserve base value
    is_public: bool = False


def build_pod_template_payload(base: dict, overrides: PodTemplateOverrides) -> dict:
    """Pure function: given the live base template dict and the explicit
    overrides above, returns the POST /v1/templates payload for the new
    template. No network calls -- kept separate from `create_pod_template`
    so the exact payload shape is directly unit-testable."""
    payload: dict = {
        "name": overrides.name,
        "imageName": overrides.image,
        "isServerless": False,  # this is a Pod template, not a Serverless one
        "isPublic": overrides.is_public,
        "category": base.get("category", "NVIDIA"),
        "containerDiskInGb": (
            overrides.container_disk_gb if overrides.container_disk_gb is not None else base.get("containerDiskInGb")
        ),
        "volumeInGb": base.get("volumeInGb"),
        "volumeMountPath": base.get("volumeMountPath"),
        "ports": _normalize_ports(base.get("ports"), overrides.required_ports),
        "env": {**(base.get("env") or {}), **overrides.env_overrides},
        # Registry auth is orthogonal to the image swap -- preserved
        # verbatim from the base template. `startSsh`/`startJupyter` are
        # NOT included here: confirmed live that RunPod's POST
        # /v1/templates schema rejects them outright ("Extra input keys
        # provided in request body ... not in input schema") even though
        # GET /v1/templates echoes them back on read -- they are
        # output-only on this endpoint, not settable at creation time.
        # Whatever value the new template ends up with is RunPod's own
        # creation default, not something this function can control.
        "containerRegistryAuthId": base.get("containerRegistryAuthId", ""),
    }
    if overrides.start_command is not None:
        payload["dockerStartCmd"] = overrides.start_command
    elif base.get("dockerStartCmd"):
        payload["dockerStartCmd"] = base["dockerStartCmd"]
    return payload


def create_pod_template(
    transport: Transport,
    api_key: str,
    *,
    base: dict,
    overrides: PodTemplateOverrides,
    log: LogFn = _default_log,
) -> tuple[Optional[dict], Optional[str]]:
    """POST /v1/templates. Returns (template_dict, None) on success or
    (None, error_detail) on any non-2xx response -- never raises, never
    silently retries with a different payload."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = build_pod_template_payload(base, overrides)
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/templates", headers=headers, json_body=payload)
    log(
        OrchestrationEvent(
            "pod_template_create_attempt",
            time.time(),
            {"name": overrides.name, "status_code": resp.status_code},
        )
    )
    if resp.status_code in (200, 201):
        return resp.json_body or {}, None
    return None, f"http {resp.status_code}: {resp.json_body}"
