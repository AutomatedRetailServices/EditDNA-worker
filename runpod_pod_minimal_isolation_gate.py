"""RunPod Pod MINIMAL EXECUTION ISOLATION test (D-042 follow-up: "FINAL
POD EXECUTION ISOLATION -- MINIMAL KNOWN-GOOD IMAGE").

Three live Pods across two cloud types and two execution transports --
all running the CutSell image -- have hit the identical `machine: {}`
empty-record signature:
  - COMMUNITY cloud + HTTP transport   (l368986gtg5ijn)
  - COMMUNITY cloud + direct execution (aejb4hkhegwpk5)
  - SECURE cloud + direct execution    (u1nftzx1i1lrik)

This isolates the one remaining shared variable across all three: is the
failure in the CutSell image/runtime itself, or in RunPod's account/host
execution layer for this account -- which would show up regardless of
image?

Deliberately uses a minimal, public, non-CutSell image (default:
`nvidia/cuda:12.4.1-base-ubuntu22.04`, the official NVIDIA CUDA base
image -- minimal, small, and the most defensible "public, CUDA-
compatible" choice available without external access to RunPod's own
usage telemetry) and a trivial shell command that needs nothing CutSell-
specific: no S3, no AWS/Gemini/Redis credentials, no pod_job_server, no
port 8080, no Video00. The only question this script answers: did the
container process actually execute?

Readiness/completion cannot be observed via an S3 marker here -- removing
every CutSell-specific dependency is the whole point -- so this script
takes bounded snapshots of the Pod's own GET state instead, watching
`machine` populate (or not) over a fixed observation window, plus a
best-effort container-log fetch. This account's log endpoints have
403/400'd on every prior Pod regardless of image; their absence here is
not itself evidence either way -- only a populated `machine` record or
actual log content confirming execution counts as positive evidence.

ALWAYS stops AND deletes the Pod in `finally` -- unlike the CutSell-Pod-
QA identity the other D-042 scripts intentionally keep alive for reuse,
this is a one-off ad hoc image with no reuse story.

Infrastructure/orchestration only -- no CutSell editorial logic lives
here, and none is imported.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from runpod_orchestration import UrllibTransport, _default_log
from runpod_pod_provider import (
    DEFAULT_COST_CEILING_USD_PER_HR,
    POD_CLOUD_TYPES,
    PodExecutionConfig,
    RunPodPodExecutionProvider,
    delete_pod,
    fetch_pod_logs,
    get_pod,
)
from runpod_pod_template import redact_template_env

DEFAULT_ISOLATION_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"
DEFAULT_ISOLATION_START_COMMAND = "sh -c 'echo POD_EXECUTION_OK; sleep 60'"


def build_minimal_isolation_config(
    *,
    api_key: str,
    pod_name: str,
    image: str,
    start_command: str,
    container_disk_gb: int,
    cost_ceiling_usd_per_hr: float,
    cloud_types: tuple[str, ...],
) -> PodExecutionConfig:
    """Pure function: no template, no env vars, no CutSell dependency of
    any kind -- everything this Pod needs is baked into `image` and
    `start_command`."""
    return PodExecutionConfig(
        api_key=api_key,
        image=image,
        pod_name=pod_name,
        start_command=start_command,
        container_disk_gb=container_disk_gb,
        env=None,
        cost_ceiling_usd_per_hr=cost_ceiling_usd_per_hr,
        cloud_types=cloud_types,
    )


def collect_pod_snapshots(
    transport,
    api_key: str,
    pod_id: str,
    *,
    window_s: float,
    interval_s: float,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    get_pod_fn=get_pod,
) -> list[dict]:
    """Bounded observation: snapshots the Pod's own GET state every
    `interval_s` for `window_s`, redacting any env values (this image
    should never have any -- stays consistent with the blanket redaction
    policy regardless). Always returns at least one snapshot, even when
    window_s < interval_s. A single failed GET is recorded and does not
    stop the loop -- a transient API error here is not itself evidence
    about container execution."""
    start = now()
    snapshots: list[dict] = []
    while True:
        try:
            pod = get_pod_fn(transport, api_key, pod_id)
            snapshot = {
                "elapsed_s": now() - start,
                "pod": redact_template_env(pod) if pod is not None else None,
            }
        except RuntimeError as exc:
            snapshot = {"elapsed_s": now() - start, "error": str(exc)}
        snapshots.append(snapshot)
        if now() - start >= window_s:
            return snapshots
        sleep(interval_s)


def machine_ever_populated(snapshots: list[dict]) -> bool:
    """True iff any snapshot's Pod state shows a non-empty `machine`
    record -- the strongest available signal (short of stdout) that this
    Pod actually landed on real, attached compute."""
    for snap in snapshots:
        pod = snap.get("pod")
        if isinstance(pod, dict) and pod.get("machine"):
            return True
    return False


def log_confirms_execution(safe_body: object) -> bool:
    """True iff the (already-redacted) log-fetch body contains the
    sentinel our start command prints. Best-effort only -- this account's
    log endpoints have never answered for any Pod tested so far, so a
    False here is not itself evidence of anything."""
    if isinstance(safe_body, str):
        return "POD_EXECUTION_OK" in safe_body
    if isinstance(safe_body, dict):
        return "POD_EXECUTION_OK" in json.dumps(safe_body)
    return False


def main() -> int:
    api_key = os.environ["RUNPOD_API_KEY"]
    existing_pod_id = os.environ.get("EXISTING_POD_ID") or None
    cost_ceiling = float(os.environ.get("QA_POD_COST_CEILING_USD_PER_HR", str(DEFAULT_COST_CEILING_USD_PER_HR)))
    image = os.environ.get("POD_ISOLATION_IMAGE") or DEFAULT_ISOLATION_IMAGE
    start_command = os.environ.get("POD_ISOLATION_START_COMMAND") or DEFAULT_ISOLATION_START_COMMAND
    container_disk_gb = int(os.environ.get("POD_ISOLATION_CONTAINER_DISK_GB", "20"))
    observe_window_s = float(os.environ.get("OBSERVE_WINDOW_S", "90"))
    observe_poll_interval_s = float(os.environ.get("OBSERVE_POLL_INTERVAL_S", "15"))

    # This isolation test is specifically authorized for SECURE cloud
    # (per the standing directive); still validated/overridable rather
    # than silently trusting a typo'd env var.
    cloud_type = (os.environ.get("POD_ISOLATION_CLOUD_TYPE") or "SECURE").strip().upper()
    if cloud_type not in POD_CLOUD_TYPES:
        print(
            f"POD_ISOLATION_CLOUD_TYPE={cloud_type!r} is not one of {POD_CLOUD_TYPES} -- refusing to guess. Aborting.",
            flush=True,
        )
        return 1

    summary: dict = {
        "image": image,
        "start_command": start_command,
        "cloud_types_requested": [cloud_type],
    }

    transport = UrllibTransport()
    config = build_minimal_isolation_config(
        api_key=api_key,
        pod_name=os.environ.get("POD_NAME", "cutsell-qa-pod-minimal-isolation"),
        image=image,
        start_command=start_command,
        container_disk_gb=container_disk_gb,
        cost_ceiling_usd_per_hr=cost_ceiling,
        cloud_types=(cloud_type,),
    )
    provider = RunPodPodExecutionProvider(transport, config, existing_pod_id=existing_pod_id, log=_default_log)

    pod_id: Optional[str] = None
    try:
        lifecycle = provider.ensure_ready()
        pod_id = lifecycle.pod_id
        summary["pod_id"] = pod_id
        summary["lifecycle_classification"] = lifecycle.classification
        if not pod_id:
            summary["classification"] = "POD_LIFECYCLE_FAILED"
            print("Pod lifecycle failed before any container could run.", flush=True)
            return 1

        print(f"--- [pod-isolation] observing Pod {pod_id} state for {observe_window_s}s ---", flush=True)
        snapshots = collect_pod_snapshots(
            transport,
            api_key,
            pod_id,
            window_s=observe_window_s,
            interval_s=observe_poll_interval_s,
            # Explicit, not relying on collect_pod_snapshots' own default --
            # that default is bound once at import time, so a test
            # monkeypatching this module's `get_pod` name would otherwise
            # never reach it. This lookup happens at call time instead.
            get_pod_fn=get_pod,
        )
        summary["snapshots"] = snapshots
        machine_populated = machine_ever_populated(snapshots)
        summary["machine_ever_populated"] = machine_populated
        print(f"--- [pod-isolation] machine_ever_populated={machine_populated} ---", flush=True)

        url, status_code, body = fetch_pod_logs(transport, api_key, pod_id, log=_default_log)
        safe_body = redact_template_env(body) if isinstance(body, dict) else body
        summary["logs_fetch"] = {"url": url, "status_code": status_code, "body": safe_body}
        print(f"--- [pod-isolation] logs fetch: {url} -> http {status_code} ---", flush=True)
        print(json.dumps(safe_body, indent=2, default=str) if safe_body is not None else "(no body)", flush=True)

        execution_confirmed_by_logs = log_confirms_execution(safe_body)
        summary["log_confirms_execution"] = execution_confirmed_by_logs

        if machine_populated or execution_confirmed_by_logs:
            summary["classification"] = "CONTAINER_EXECUTION_CONFIRMED"
        else:
            summary["classification"] = "CONTAINER_EXECUTION_NOT_CONFIRMED"

        return 0
    finally:
        # Guaranteed cleanup: STOP via the shared provider (same
        # guarantee every other D-042 script relies on), then DELETE --
        # this ad hoc image has no reuse story, unlike the CutSell-Pod-QA
        # identity the other scripts deliberately keep alive.
        provider.teardown()
        if pod_id:
            deleted = delete_pod(transport, api_key, pod_id)
            summary["pod_deleted"] = deleted
            print(f"--- [pod-isolation] delete_pod({pod_id}) -> {deleted} ---", flush=True)
        Path("pod-minimal-isolation-summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print("--- pod-minimal-isolation-summary.json ---", flush=True)
        print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
