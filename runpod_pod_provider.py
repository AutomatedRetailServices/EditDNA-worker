"""RunPod Pod on-demand execution provider (D-042: CutSell QA GPU execution
fallback -- RunPod Pod On-Demand automation. KEEP SERVERLESS FULLY
AVAILABLE; this is an EXECUTION/INFRASTRUCTURE addition only).

Serverless (`gpu_execution_provider.RunPodServerlessExecutionProvider`)
remains the production backend, untouched. This module adds
`RunPodPodExecutionProvider`, a durable QA Pod lifecycle: TEST REQUESTED ->
inspect QA Pod state -> reuse a compatible stopped Pod if possible -> START
-> wait ready -> health -> (full benchmark, not yet authorized) -> STOP Pod
in finally/always, regardless of outcome.

Every RunPod HTTP call goes through the same injectable `Transport`
protocol `runpod_orchestration.py` already defines -- this module runs
entirely against fakes in tests, no paid GPU or RunPod credentials
required to validate the state machine itself. Reuses that module's
Transport/OrchestrationEvent primitives rather than redefining them.

GPU compatibility (current image: PyTorch 2.6 / CUDA 12.4): approved pool
is RTX 4090, A40, RTX A6000, L4, in that conceptual preference order.
NVIDIA RTX PRO 6000 Blackwell Server Edition, any other Blackwell part
(sm_120), and any GPU outside this pool (H100, H200, A100 included) are
never provisioned automatically by this module, regardless of price or
availability, until a separate PyTorch/CUDA upgrade project validates
them.

Concurrency: two QA workflows must never provision/run against the same
Pod simultaneously. This module deliberately does not implement its own
distributed lock -- the repo already has an officially-supported
mechanism for exactly this (see `cutsell-video00-raw-v5-auto-microtrim.
yml`'s `concurrency: group: ...` block); the new Pod QA workflow uses the
same GitHub Actions concurrency-group mechanism instead of a second,
divergent locking primitive. See `.github/workflows/cutsell-video00-pod-raw.yml`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from runpod_orchestration import (
    RUNPOD_REST_BASE,
    LogFn,
    OrchestrationEvent,
    Transport,
    _default_log,
)

# ---------------------------------------------------------------------------
# Approved GPU pool (D-042) -- conceptual preference order. Availability and
# cost may alter the actual selection; this order is only the starting rank.
# ---------------------------------------------------------------------------
APPROVED_POD_GPU_TYPE_IDS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA L4",
)

# Explicitly never provisioned automatically, regardless of price/availability,
# even if RunPod's own catalog lists them as available -- Blackwell (sm_120)
# is the exact incompatibility this whole D-042 body of work traces back to
# (the GPU-fallback-audit follow-up), and H100/H200/A100 are simply out of
# QA's cost/scope envelope.
EXCLUDED_POD_GPU_TYPE_IDS = frozenset(
    {
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 NVL",
        "NVIDIA H100 PCIe",
        "NVIDIA H200",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
    }
)

DEFAULT_COST_CEILING_USD_PER_HR = 1.50

# Pod-lifecycle classification vocabulary, mirroring runpod_orchestration.py's
# style so both backends' logs read the same way.
POD_REUSED = "POD_REUSED"
POD_RESTARTED = "POD_RESTARTED"
POD_CREATED_FRESH = "POD_CREATED_FRESH"
POD_STALE_RECREATED = "POD_STALE_RECREATED"
POD_RESTART_UNAVAILABLE = "POD_RESTART_UNAVAILABLE"
POD_CAPACITY_UNAVAILABLE = "POD_CAPACITY_UNAVAILABLE"
POD_COST_CEILING_EXCEEDED = "POD_COST_CEILING_EXCEEDED"
POD_GPU_OUTSIDE_ALLOWLIST = "POD_GPU_OUTSIDE_ALLOWLIST"
POD_RUNPOD_API_ERROR = "POD_RUNPOD_API_ERROR"
POD_HEALTH_APP_FAILURE = "POD_HEALTH_APP_FAILURE"
POD_HEALTH_PASSED = "POD_HEALTH_PASSED"
POD_OWNED_BY_OTHER_RUN = "POD_OWNED_BY_OTHER_RUN"

_TERMINAL_READY_STATUSES = frozenset({"RUNNING"})
_STOPPED_STATUSES = frozenset({"EXITED", "STOPPED"})


# ---------------------------------------------------------------------------
# GPU catalog + selection (D-042: GPU availability search + cost safety)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUCandidate:
    gpu_type_id: str
    price_usd_per_hr: Optional[float]
    available: Optional[bool]  # None when the catalog doesn't say either way


@dataclass(frozen=True)
class GPUSelection:
    chosen: Optional[GPUCandidate]
    candidates_considered: tuple[GPUCandidate, ...]
    classification: Optional[str]  # set only when chosen is None
    reason: str


def fetch_pod_gpu_catalog(transport: Transport, api_key: str, *, log: LogFn = _default_log) -> dict[str, dict]:
    """Read-only GET against RunPod's GPU catalog. Never assumes one GPU
    must be available -- returns whatever the catalog reports (price and,
    when present, an availability signal), defensively handling multiple
    possible field-name shapes the same way `runpod_fast_isolation_probe.
    resolve_current_gpu_type_ids` already does. Returns {} (not an
    exception) on any catalog-read failure -- callers fall back to
    attempt-based selection (see `select_and_create_pod`)."""
    headers = {"Authorization": f"Bearer {api_key}"}
    for url in (f"{RUNPOD_REST_BASE}/gpuTypes", f"{RUNPOD_REST_BASE}/gpu-types"):
        resp = transport.request("GET", url, headers=headers)
        if resp.status_code == 200 and isinstance(resp.json_body, list):
            catalog: dict[str, dict] = {}
            for entry in resp.json_body:
                if not isinstance(entry, dict):
                    continue
                gpu_id = str(entry.get("id") or "")
                if not gpu_id:
                    continue
                lowest_price = entry.get("lowestPrice")
                lowest_price_value = (
                    lowest_price.get("uninterruptablePrice") if isinstance(lowest_price, dict) else None
                )
                price = entry.get("communityPrice") or entry.get("securePrice") or lowest_price_value
                available = entry.get("communityCloud")
                if available is None:
                    available = entry.get("secureCloud")
                catalog[gpu_id] = {
                    "price_usd_per_hr": float(price) if isinstance(price, (int, float)) else None,
                    "available": bool(available) if isinstance(available, bool) else None,
                }
            log(OrchestrationEvent("pod_gpu_catalog_fetched", time.time(), {"catalog_url": url, "count": len(catalog)}))
            return catalog
    log(OrchestrationEvent("pod_gpu_catalog_unavailable", time.time(), {}))
    return {}


def rank_gpu_candidates(
    catalog: dict[str, dict],
    *,
    cost_ceiling_usd_per_hr: float,
    approved_ids: tuple[str, ...] = APPROVED_POD_GPU_TYPE_IDS,
) -> GPUSelection:
    """Rank the approved pool by (1) compatibility -- already enforced by
    only ever looking at `approved_ids`/never `EXCLUDED_POD_GPU_TYPE_IDS`,
    (2) availability, (3) conceptual preference order (proxy for expected
    performance), (4) cost ceiling. Returns the first approved candidate
    the catalog reports as available and under the cost ceiling; if the
    catalog has no opinion on availability for a candidate (`None`), it is
    still considered eligible (attempt-based creation is the real
    availability test -- see `select_and_create_pod`), but a candidate the
    catalog explicitly reports unavailable (`False`) is skipped.
    """
    considered: list[GPUCandidate] = []
    for gpu_id in approved_ids:
        assert gpu_id not in EXCLUDED_POD_GPU_TYPE_IDS, f"{gpu_id} must never be in the approved pool"
        entry = catalog.get(gpu_id) or {}
        considered.append(
            GPUCandidate(
                gpu_type_id=gpu_id,
                price_usd_per_hr=entry.get("price_usd_per_hr"),
                available=entry.get("available"),
            )
        )

    eligible = [c for c in considered if c.available is not False]
    under_ceiling = [
        c for c in eligible if c.price_usd_per_hr is None or c.price_usd_per_hr <= cost_ceiling_usd_per_hr
    ]
    over_ceiling_only = eligible and not under_ceiling

    if not eligible:
        return GPUSelection(
            chosen=None,
            candidates_considered=tuple(considered),
            classification=POD_CAPACITY_UNAVAILABLE,
            reason="No approved GPU type was reported available by the catalog.",
        )
    if over_ceiling_only:
        return GPUSelection(
            chosen=None,
            candidates_considered=tuple(considered),
            classification=POD_COST_CEILING_EXCEEDED,
            reason=(
                f"Every available approved GPU type is priced above the "
                f"${cost_ceiling_usd_per_hr:.2f}/hr QA cost ceiling."
            ),
        )
    chosen = under_ceiling[0]
    return GPUSelection(
        chosen=chosen,
        candidates_considered=tuple(considered),
        classification=None,
        reason=(
            f"Selected {chosen.gpu_type_id} (${chosen.price_usd_per_hr}/hr) -- "
            f"first approved, available, under-ceiling candidate in preference order."
        ),
    )


# ---------------------------------------------------------------------------
# Pod REST primitives
# ---------------------------------------------------------------------------
def get_pod(transport: Transport, api_key: str, pod_id: str) -> Optional[dict]:
    """Returns the pod's live state dict, or None if it no longer exists
    (404) -- never guesses at existence."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = transport.request("GET", f"{RUNPOD_REST_BASE}/pods/{pod_id}", headers=headers)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"GET pod {pod_id} failed: http {resp.status_code}")
    return resp.json_body or {}


def create_pod(
    transport: Transport,
    api_key: str,
    *,
    name: str,
    image: str,
    gpu_type_id: str,
    start_command: Optional[str] = None,
    container_disk_gb: int = 40,
    env: Optional[dict] = None,
    ports: str = "8080/http",
) -> tuple[Optional[dict], Optional[str]]:
    """POST /v1/pods. Returns (pod_dict, None) on success or (None,
    error_detail) on any non-2xx response -- the caller decides whether that
    detail looks capacity-shaped (try the next candidate) or fatal."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: dict = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu_type_id],
        "gpuCount": 1,
        "containerDiskInGb": container_disk_gb,
        "cloudType": "COMMUNITY",
        "ports": ports,
        "env": env or {},
    }
    if start_command:
        payload["dockerStartCmd"] = start_command
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/pods", headers=headers, json_body=payload)
    if resp.status_code in (200, 201):
        return resp.json_body or {}, None
    detail = f"http {resp.status_code}: {resp.json_body}"
    return None, detail


_CAPACITY_ERROR_MARKERS = (
    "no instances",
    "not available",
    "no longer available",
    "insufficient capacity",
    "out of stock",
    "no gpu",
    "unavailable",
)


def looks_like_capacity_error(detail: str) -> bool:
    lowered = (detail or "").lower()
    return any(marker in lowered for marker in _CAPACITY_ERROR_MARKERS)


def start_pod(transport: Transport, api_key: str, pod_id: str) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/pods/{pod_id}/start", headers=headers, json_body={})
    return resp.status_code in (200, 201)


def stop_pod(transport: Transport, api_key: str, pod_id: str) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = transport.request("POST", f"{RUNPOD_REST_BASE}/pods/{pod_id}/stop", headers=headers, json_body={})
    return resp.status_code in (200, 201)


def delete_pod(transport: Transport, api_key: str, pod_id: str) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = transport.request("DELETE", f"{RUNPOD_REST_BASE}/pods/{pod_id}", headers=headers)
    return resp.status_code in (200, 202, 204)


@dataclass(frozen=True)
class PodReadiness:
    ready: bool
    desired_status: Optional[str]
    elapsed_s: float
    classification: Optional[str] = None


def wait_for_pod_running(
    transport: Transport,
    api_key: str,
    pod_id: str,
    *,
    timeout_s: float = 180.0,
    poll_interval_s: float = 5.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: LogFn = _default_log,
) -> PodReadiness:
    """Poll GET /v1/pods/{id} until `desiredStatus`/`status` reports
    RUNNING, or `timeout_s` is spent trying. A pod that disappears (404)
    mid-wait or reports an ERROR-shaped status is classified as restart-
    unavailable rather than silently retried forever."""
    start = now()
    while True:
        elapsed = now() - start
        pod = get_pod(transport, api_key, pod_id)
        if pod is None:
            log(OrchestrationEvent("pod_missing_during_wait", now(), {"pod_id": pod_id, "elapsed_s": elapsed}))
            return PodReadiness(ready=False, desired_status=None, elapsed_s=elapsed, classification=POD_RESTART_UNAVAILABLE)
        status = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
        log(OrchestrationEvent("pod_wait_status", now(), {"pod_id": pod_id, "status": status, "elapsed_s": elapsed}))
        if status in _TERMINAL_READY_STATUSES:
            return PodReadiness(ready=True, desired_status=status, elapsed_s=elapsed)
        if status in {"ERROR", "TERMINATED", "BAD_REQUEST"}:
            return PodReadiness(ready=False, desired_status=status, elapsed_s=elapsed, classification=POD_RESTART_UNAVAILABLE)
        if elapsed >= timeout_s:
            return PodReadiness(ready=False, desired_status=status, elapsed_s=elapsed, classification=POD_RESTART_UNAVAILABLE)
        sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# High-level lifecycle provider
# ---------------------------------------------------------------------------
@dataclass
class PodExecutionConfig:
    api_key: str
    image: str
    pod_name: str = "cutsell-qa-pod"
    start_command: Optional[str] = None
    container_disk_gb: int = 40
    cost_ceiling_usd_per_hr: float = DEFAULT_COST_CEILING_USD_PER_HR
    approved_gpu_type_ids: tuple[str, ...] = APPROVED_POD_GPU_TYPE_IDS
    health_port: int = 8080
    restart_wait_timeout_s: float = 180.0
    poll_interval_s: float = 5.0


@dataclass(frozen=True)
class PodLifecycleResult:
    pod_id: Optional[str]
    classification: str  # one of the POD_* constants above
    gpu_selection: Optional[GPUSelection]
    elapsed_s: float
    detail: dict = field(default_factory=dict)


class RunPodPodExecutionProvider:
    """Implements `gpu_execution_provider.GPUExecutionProvider` for RunPod
    Pods. `existing_pod_id`, when known (e.g. persisted between workflow
    runs via a repo variable or the caller's own state file), is the
    "one optional QA Pod identity" the directive asks this module to
    maintain -- reuse-first, recreate-on-failure, never unbounded retry."""

    def __init__(
        self,
        transport: Transport,
        config: PodExecutionConfig,
        *,
        existing_pod_id: Optional[str] = None,
        http_get: Optional[Callable[[str], tuple[int, Optional[dict]]]] = None,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: LogFn = _default_log,
    ) -> None:
        self._transport = transport
        self._cfg = config
        self._existing_pod_id = existing_pod_id
        self._http_get = http_get  # injected for tests; production uses urllib against the Pod's proxy URL
        self._now = now
        self._sleep = sleep
        self._log = log
        self._pod_id: Optional[str] = None
        self._lifecycle_classification: Optional[str] = None

    @property
    def pod_id(self) -> Optional[str]:
        """The Pod this instance is currently holding (after `ensure_ready`/
        `health_check`), or the caller-supplied `existing_pod_id` if neither
        has run yet. `None` before anything has run and no identity was
        supplied."""
        return self._pod_id or self._existing_pod_id

    # -- GPU search + fresh creation -----------------------------------
    def _select_and_create_fresh(self) -> PodLifecycleResult:
        start = self._now()
        catalog = fetch_pod_gpu_catalog(self._transport, self._cfg.api_key, log=self._log)
        selection = rank_gpu_candidates(
            catalog,
            cost_ceiling_usd_per_hr=self._cfg.cost_ceiling_usd_per_hr,
            approved_ids=self._cfg.approved_gpu_type_ids,
        )
        # Attempt-based fallback through the ranked, under-ceiling candidates
        # in order -- the catalog's own `available` flag is advisory; a real
        # creation attempt is the authoritative availability signal, exactly
        # like a Pod-capacity check should be (there is no separate
        # "reserve capacity" endpoint to pre-check against).
        ordered = [c for c in selection.candidates_considered if c.available is not False]
        ordered = [
            c for c in ordered if c.price_usd_per_hr is None or c.price_usd_per_hr <= self._cfg.cost_ceiling_usd_per_hr
        ]
        if not ordered:
            self._log(OrchestrationEvent("pod_no_eligible_gpu_candidate", self._now(), {"reason": selection.reason}))
            return PodLifecycleResult(
                pod_id=None,
                classification=selection.classification or POD_CAPACITY_UNAVAILABLE,
                gpu_selection=selection,
                elapsed_s=self._now() - start,
                detail={"reason": selection.reason, "candidates": [c.gpu_type_id for c in selection.candidates_considered]},
            )

        last_detail = None
        for candidate in ordered:
            pod, detail = create_pod(
                self._transport,
                self._cfg.api_key,
                name=self._cfg.pod_name,
                image=self._cfg.image,
                gpu_type_id=candidate.gpu_type_id,
                start_command=self._cfg.start_command,
                container_disk_gb=self._cfg.container_disk_gb,
            )
            if pod is not None:
                pod_id = str(pod.get("id") or "")
                self._log(
                    OrchestrationEvent(
                        "pod_created_fresh",
                        self._now(),
                        {"pod_id": pod_id, "gpu_type_id": candidate.gpu_type_id, "price_usd_per_hr": candidate.price_usd_per_hr},
                    )
                )
                return PodLifecycleResult(
                    pod_id=pod_id,
                    classification=POD_CREATED_FRESH,
                    gpu_selection=GPUSelection(
                        chosen=candidate,
                        candidates_considered=selection.candidates_considered,
                        classification=None,
                        reason=f"Created with {candidate.gpu_type_id}.",
                    ),
                    elapsed_s=self._now() - start,
                    detail={"gpu_type_id": candidate.gpu_type_id},
                )
            last_detail = detail
            if not looks_like_capacity_error(detail or ""):
                # A non-capacity-shaped failure (auth, malformed request,
                # image pull, quota) is fatal -- do not keep guessing GPUs.
                self._log(OrchestrationEvent("pod_create_fatal_error", self._now(), {"gpu_type_id": candidate.gpu_type_id, "detail": detail}))
                return PodLifecycleResult(
                    pod_id=None,
                    classification=POD_RUNPOD_API_ERROR,
                    gpu_selection=selection,
                    elapsed_s=self._now() - start,
                    detail={"gpu_type_id": candidate.gpu_type_id, "error": detail},
                )
            self._log(OrchestrationEvent("pod_create_capacity_unavailable_trying_next", self._now(), {"gpu_type_id": candidate.gpu_type_id, "detail": detail}))

        return PodLifecycleResult(
            pod_id=None,
            classification=POD_CAPACITY_UNAVAILABLE,
            gpu_selection=selection,
            elapsed_s=self._now() - start,
            detail={"reason": "Every candidate GPU rejected pod creation with a capacity-shaped error.", "last_error": last_detail},
        )

    # -- Reuse / restart / stale recovery -------------------------------
    def ensure_ready(self) -> PodLifecycleResult:
        """TEST REQUESTED -> inspect existing Pod -> reuse if possible ->
        one bounded restart attempt -> stale? delete + create fresh ->
        none existed? create fresh. Never loops unbounded on the same Pod."""
        start = self._now()
        if self._existing_pod_id:
            try:
                pod = get_pod(self._transport, self._cfg.api_key, self._existing_pod_id)
            except RuntimeError as exc:
                self._log(OrchestrationEvent("pod_inspect_error", self._now(), {"pod_id": self._existing_pod_id, "error": str(exc)}))
                pod = None

            if pod is not None:
                image_matches = str(pod.get("imageName") or "") == self._cfg.image
                status = str(pod.get("desiredStatus") or pod.get("status") or "").upper()

                if status in _TERMINAL_READY_STATUSES and image_matches:
                    self._pod_id = self._existing_pod_id
                    self._log(OrchestrationEvent("pod_reused_already_running", self._now(), {"pod_id": self._pod_id}))
                    return PodLifecycleResult(
                        pod_id=self._pod_id, classification=POD_REUSED, gpu_selection=None, elapsed_s=self._now() - start,
                    )

                if status in _STOPPED_STATUSES and image_matches:
                    self._log(OrchestrationEvent("pod_restart_attempt", self._now(), {"pod_id": self._existing_pod_id}))
                    started = start_pod(self._transport, self._cfg.api_key, self._existing_pod_id)
                    if started:
                        readiness = wait_for_pod_running(
                            self._transport,
                            self._cfg.api_key,
                            self._existing_pod_id,
                            timeout_s=self._cfg.restart_wait_timeout_s,
                            poll_interval_s=self._cfg.poll_interval_s,
                            now=self._now,
                            sleep=self._sleep,
                            log=self._log,
                        )
                        if readiness.ready:
                            self._pod_id = self._existing_pod_id
                            return PodLifecycleResult(
                                pod_id=self._pod_id, classification=POD_RESTARTED, gpu_selection=None, elapsed_s=self._now() - start,
                            )
                    # Restart failed or never became ready within the bounded
                    # timeout -- stop waiting, delete the stale Pod, fall
                    # through to fresh creation. Exactly one recreate
                    # attempt; no loop.
                    self._log(OrchestrationEvent("pod_restart_unavailable_deleting_stale", self._now(), {"pod_id": self._existing_pod_id}))
                    delete_pod(self._transport, self._cfg.api_key, self._existing_pod_id)
                    result = self._select_and_create_fresh()
                    if result.pod_id:
                        self._pod_id = result.pod_id
                        return PodLifecycleResult(
                            pod_id=result.pod_id,
                            classification=POD_STALE_RECREATED,
                            gpu_selection=result.gpu_selection,
                            elapsed_s=self._now() - start,
                            detail=result.detail,
                        )
                    return result

                # RUNNING-but-wrong-image, ERROR, or any other unexpected
                # state with no clean reuse/restart path -- treat as stale:
                # delete and recreate rather than colliding with whatever
                # state it's actually in.
                self._log(OrchestrationEvent("pod_stale_state_deleting", self._now(), {"pod_id": self._existing_pod_id, "status": status, "image_matches": image_matches}))
                delete_pod(self._transport, self._cfg.api_key, self._existing_pod_id)
                result = self._select_and_create_fresh()
                if result.pod_id:
                    self._pod_id = result.pod_id
                    return PodLifecycleResult(
                        pod_id=result.pod_id,
                        classification=POD_STALE_RECREATED,
                        gpu_selection=result.gpu_selection,
                        elapsed_s=self._now() - start,
                        detail=result.detail,
                    )
                return result

            # existing_pod_id was set but the Pod no longer exists (MISSING).
            self._log(OrchestrationEvent("pod_missing_creating_fresh", self._now(), {"pod_id": self._existing_pod_id}))

        result = self._select_and_create_fresh()
        if result.pod_id:
            self._pod_id = result.pod_id
        return result

    # -- GPUExecutionProvider interface ---------------------------------
    def health_check(self):
        from gpu_execution_provider import HealthCheckResult  # local import avoids a cycle at module load

        lifecycle = self.ensure_ready()
        if not lifecycle.pod_id:
            return HealthCheckResult(
                execution_provider="RUNPOD_POD",
                passed=False,
                classification=lifecycle.classification,
                elapsed_s=lifecycle.elapsed_s,
                detail=lifecycle.detail,
            )

        if self._http_get is not None:
            status_code, body = self._http_get(self._pod_health_url())
        else:
            status_code, body = self._real_http_get(self._pod_health_url())

        passed = status_code == 200 and isinstance(body, dict) and body.get("ok") is True and body.get("cuda_available") is True
        classification = POD_HEALTH_PASSED if passed else POD_HEALTH_APP_FAILURE
        return HealthCheckResult(
            execution_provider="RUNPOD_POD",
            passed=passed,
            classification=classification,
            elapsed_s=lifecycle.elapsed_s,
            detail={
                "pod_id": self._pod_id,
                "lifecycle_classification": lifecycle.classification,
                "gpu_type_id": lifecycle.gpu_selection.chosen.gpu_type_id if (lifecycle.gpu_selection and lifecycle.gpu_selection.chosen) else None,
                "price_usd_per_hr": lifecycle.gpu_selection.chosen.price_usd_per_hr if (lifecycle.gpu_selection and lifecycle.gpu_selection.chosen) else None,
                "health_status_code": status_code,
                "health_payload": body,
            },
        )

    def _pod_health_url(self) -> str:
        # RunPod's documented HTTP proxy convention for an exposed Pod port.
        return f"https://{self._pod_id}-{self._cfg.health_port}.proxy.runpod.net/health"

    def _real_http_get(self, url: str) -> tuple[int, Optional[dict]]:
        import json
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except ValueError:
                return exc.code, None
        except Exception as exc:  # noqa: BLE001 -- network errors must not crash the health gate
            self._log(OrchestrationEvent("pod_health_http_error", self._now(), {"error": str(exc)}))
            return 0, None

    def teardown(self) -> None:
        """Guaranteed-cleanup: STOP (never delete -- reuse-first policy)
        the Pod this instance is holding, if any. Retries once on failure,
        then logs an escalation event rather than silently leaving a paid
        GPU running. Safe to call even if `ensure_ready`/`health_check`
        never ran or failed."""
        pod_id = self._pod_id or self._existing_pod_id
        if not pod_id:
            self._log(OrchestrationEvent("pod_teardown_skipped_no_pod", self._now(), {}))
            return
        ok = stop_pod(self._transport, self._cfg.api_key, pod_id)
        if not ok:
            self._log(OrchestrationEvent("pod_stop_retry", self._now(), {"pod_id": pod_id}))
            ok = stop_pod(self._transport, self._cfg.api_key, pod_id)
        if not ok:
            self._log(OrchestrationEvent("pod_stop_failed_escalate", self._now(), {"pod_id": pod_id}))
            return
        pod_after = None
        try:
            pod_after = get_pod(self._transport, self._cfg.api_key, pod_id)
        except RuntimeError:
            pass
        final_status = str((pod_after or {}).get("desiredStatus") or (pod_after or {}).get("status") or "unknown")
        self._log(OrchestrationEvent("pod_stopped", self._now(), {"pod_id": pod_id, "final_status": final_status}))
