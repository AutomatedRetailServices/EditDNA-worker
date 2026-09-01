"""RunPod endpoint startup/health orchestration for the Video00 RAW workflow.

Infrastructure/orchestration only -- no CutSell editorial logic lives here.
See docs/CUTSELL_DECISIONS.md D-041.

RAW `33453836301` spent its entire 20-minute "CUDA health" budget blind: it
submitted a RunPod serverless health job, polled its status every 5s against
one undifferentiated deadline, and only at the very end printed whatever the
last poll happened to show (`{"status":"IN_QUEUE"}` -- the job was accepted
but no worker ever picked it up). No application code -- including any
CutSell editorial code -- ever ran. This module replaces that single blind
wait with a small, fully unit-testable state machine that distinguishes:

  - ENDPOINT_TRANSITION_RACE   -- the endpoint itself is still mid-transition
                                   (RunPod returns 409 on read) after we rolled it
  - RUNPOD_API_ERROR           -- an unexpected HTTP-level failure talking to RunPod
  - CAPACITY_UNAVAILABLE       -- the endpoint's own config never stabilized within
                                   its bounded readiness timeout (no template/worker
                                   pool ever came up for this roll)
  - WORKER_PROVISIONING_STALLED -- the endpoint IS ready and a health job WAS
                                   accepted, but it sat in IN_QUEUE past a bounded
                                   stall threshold with no worker ever picking it up
  - HEALTH_APP_FAILURE         -- RunPod actually ran the job and it reported
                                   FAILED/TIMED_OUT/CANCELLED, or COMPLETED with
                                   ok/cuda_available not true -- a real application
                                   or CUDA problem, not a queue/capacity issue
  - HEALTH_PASSED              -- COMPLETED with ok == true and cuda_available == true

The first four are infrastructure-class failures eligible for one bounded,
non-silent retry (roll again, wait again) -- see `run_with_bounded_retry`.
HEALTH_APP_FAILURE is never retried here: it means RunPod's own worker ran
and reported a real problem, which is a signal to investigate, not a flake.

Every RunPod HTTP call goes through the `Transport` protocol below so this
whole module runs against fakes in tests -- no paid GPU, no network, no
RunPod credentials required to validate the state machine itself.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

# ---------------------------------------------------------------------------
# Failure/outcome vocabulary (D-041 item 5's required explicit classes)
# ---------------------------------------------------------------------------
ENDPOINT_TRANSITION_RACE = "ENDPOINT_TRANSITION_RACE"
CAPACITY_UNAVAILABLE = "CAPACITY_UNAVAILABLE"
WORKER_PROVISIONING_STALLED = "WORKER_PROVISIONING_STALLED"
RUNPOD_API_ERROR = "RUNPOD_API_ERROR"
HEALTH_APP_FAILURE = "HEALTH_APP_FAILURE"
HEALTH_PASSED = "HEALTH_PASSED"

# Infra-class failures are eligible for the one bounded retry; HEALTH_APP_FAILURE
# is never in this set -- an actual application/CUDA failure is not a flake.
INFRA_FAILURE_CLASSES = frozenset(
    {
        ENDPOINT_TRANSITION_RACE,
        CAPACITY_UNAVAILABLE,
        WORKER_PROVISIONING_STALLED,
        RUNPOD_API_ERROR,
    }
)

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
RUNPOD_SERVERLESS_BASE = "https://api.runpod.ai/v2"

# Terminal RunPod job statuses that mean "RunPod actually ran this" as opposed
# to "still sitting in the scheduler".
_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"})


class Transport(Protocol):
    """Injectable HTTP boundary. Production uses `RequestsTransport`; tests
    use a scripted fake -- see tests/test_runpod_orchestration.py."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict | None = None,
    ) -> "TransportResponse": ...


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    json_body: Any = None


class UrllibTransport:
    """Production Transport backed by the stdlib only (`urllib.request`) --
    deliberately no third-party dependency, matching the rest of this
    workflow's Python steps (benchmarks/validate_video00_*.py), which run
    against the GitHub-hosted runner's system Python with no pip-install
    step. Not exercised by unit tests (those inject a fake); only used from
    `main()`."""

    def __init__(self, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s

    def request(self, method, url, *, headers, json_body=None) -> TransportResponse:
        import urllib.error
        import urllib.request

        data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
        req_headers = dict(headers)
        if data is not None:
            req_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as resp:
                status_code = resp.status
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            raw = exc.read()
        except urllib.error.URLError:
            return TransportResponse(status_code=0, json_body=None)
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        return TransportResponse(status_code=status_code, json_body=body)


# ---------------------------------------------------------------------------
# Observability (D-041 item 5)
# ---------------------------------------------------------------------------
@dataclass
class OrchestrationEvent:
    name: str
    at: float
    fields: dict = field(default_factory=dict)


def _default_log(event: OrchestrationEvent) -> None:
    print(f"[runpod_orchestration] {event.name} {json.dumps(event.fields, default=str)}", flush=True)


LogFn = Callable[[OrchestrationEvent], None]


# ---------------------------------------------------------------------------
# Endpoint readiness (D-041 item 1)
# ---------------------------------------------------------------------------
@dataclass
class EndpointReadiness:
    ready: bool
    template_id: Optional[str]
    workers_min: Optional[int]
    workers_max: Optional[int]
    gpu_type: Optional[str]
    elapsed_s: float
    classification: Optional[str] = None  # set only when ready is False


def wait_for_endpoint_ready(
    transport: Transport,
    endpoint_id: str,
    api_key: str,
    expected_template_id: str,
    expected_workers_max: int,
    *,
    readiness_timeout_s: float = 180.0,
    poll_interval_s: float = 3.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: LogFn = _default_log,
) -> EndpointReadiness:
    """Poll GET /v1/endpoints/{id} until it reflects the roll we just PATCHed
    (matching templateId + workersMax), or until `readiness_timeout_s` is
    spent trying. A 409 on this read means the endpoint is still mid
    transition from the previous configuration -- distinct from any other
    HTTP-level error.
    """
    start = now()
    url = f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    saw_409 = False

    while True:
        elapsed = now() - start
        resp = transport.request("GET", url, headers=headers)

        if resp.status_code == 409:
            saw_409 = True
        elif resp.status_code == 200:
            body = resp.json_body or {}
            template_id = body.get("templateId")
            workers_min = body.get("workersMin")
            workers_max = body.get("workersMax")
            gpu_type = body.get("gpuIds") or body.get("gpuType")
            if template_id == expected_template_id and workers_max == expected_workers_max:
                log(
                    OrchestrationEvent(
                        "endpoint_ready",
                        now(),
                        {
                            "elapsed_s": elapsed,
                            "template_id": template_id,
                            "workers_min": workers_min,
                            "workers_max": workers_max,
                            "gpu_type": gpu_type,
                        },
                    )
                )
                return EndpointReadiness(
                    ready=True,
                    template_id=template_id,
                    workers_min=workers_min,
                    workers_max=workers_max,
                    gpu_type=gpu_type,
                    elapsed_s=elapsed,
                )

        if elapsed >= readiness_timeout_s:
            classification = ENDPOINT_TRANSITION_RACE if saw_409 else CAPACITY_UNAVAILABLE
            log(
                OrchestrationEvent(
                    "endpoint_not_ready",
                    now(),
                    {"elapsed_s": elapsed, "classification": classification, "last_status_code": resp.status_code},
                )
            )
            return EndpointReadiness(
                ready=False,
                template_id=None,
                workers_min=None,
                workers_max=None,
                gpu_type=None,
                elapsed_s=elapsed,
                classification=classification,
            )

        sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# Health job submission + IN_QUEUE-aware polling (D-041 item 2)
# ---------------------------------------------------------------------------
@dataclass
class HealthOutcome:
    classification: str
    passed: bool
    job_id: Optional[str]
    time_in_queue_s: float
    final_status: Optional[str]
    detail: str = ""


def submit_and_poll_health(
    transport: Transport,
    endpoint_id: str,
    api_key: str,
    *,
    queue_grace_s: float = 90.0,
    queue_stall_s: float = 300.0,
    is_retry_attempt: bool = False,
    poll_interval_s: float = 5.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: LogFn = _default_log,
) -> HealthOutcome:
    """Submit a `{"op":"health"}` job and poll its status, tracking time
    spent specifically in IN_QUEUE rather than using one undifferentiated
    deadline.

    Policy (D-041 item 2's own example, made concrete):
      - < queue_grace_s in IN_QUEUE: normal cold start, keep waiting silently.
      - >= queue_stall_s still in IN_QUEUE: fail early rather than burn the
        rest of a long blind deadline. First attempt -> WORKER_PROVISIONING_
        STALLED (treated as possibly transient, eligible for the one bounded
        retry in `run_with_bounded_retry`). Second/retry attempt still stuck
        -> CAPACITY_UNAVAILABLE (persistent; no further retries).
      - A genuine terminal RunPod status (COMPLETED/FAILED/TIMED_OUT/
        CANCELLED) is never reclassified as a queue problem, even if it took
        a while to get there -- that's HEALTH_APP_FAILURE or HEALTH_PASSED.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    submit_resp = transport.request(
        "POST",
        f"{RUNPOD_SERVERLESS_BASE}/{endpoint_id}/run",
        headers=headers,
        json_body={"input": {"op": "health"}},
    )
    if submit_resp.status_code != 200 or not (submit_resp.json_body or {}).get("id"):
        log(OrchestrationEvent("health_submit_failed", now(), {"status_code": submit_resp.status_code}))
        return HealthOutcome(
            classification=RUNPOD_API_ERROR,
            passed=False,
            job_id=None,
            time_in_queue_s=0.0,
            final_status=None,
            detail=f"submit http {submit_resp.status_code}",
        )

    job_id = submit_resp.json_body["id"]
    log(OrchestrationEvent("health_submitted", now(), {"job_id": job_id}))

    start = now()
    queue_started_at: Optional[float] = None
    last_status: Optional[str] = None

    while True:
        elapsed = now() - start
        status_resp = transport.request(
            "GET",
            f"{RUNPOD_SERVERLESS_BASE}/{endpoint_id}/status/{job_id}",
            headers=headers,
        )
        if status_resp.status_code != 200:
            log(OrchestrationEvent("health_poll_http_error", now(), {"status_code": status_resp.status_code}))
            return HealthOutcome(
                classification=RUNPOD_API_ERROR,
                passed=False,
                job_id=job_id,
                time_in_queue_s=elapsed,
                final_status=None,
                detail=f"poll http {status_resp.status_code}",
            )

        body = status_resp.json_body or {}
        status = body.get("status") or ""
        last_status = status
        worker_id = body.get("workerId") or body.get("worker_id")
        log(
            OrchestrationEvent(
                "health_status",
                now(),
                {"job_id": job_id, "status": status, "elapsed_s": elapsed, "worker_id": worker_id},
            )
        )

        if status in _TERMINAL_STATUSES:
            if status == "COMPLETED":
                output = body.get("output") or {}
                if output.get("ok") is True and output.get("cuda_available") is True:
                    return HealthOutcome(
                        classification=HEALTH_PASSED,
                        passed=True,
                        job_id=job_id,
                        time_in_queue_s=(queue_started_at is not None and (now() - queue_started_at)) or 0.0,
                        final_status=status,
                    )
            return HealthOutcome(
                classification=HEALTH_APP_FAILURE,
                passed=False,
                job_id=job_id,
                time_in_queue_s=(queue_started_at is not None and (now() - queue_started_at)) or 0.0,
                final_status=status,
                detail=json.dumps(body)[:500],
            )

        if status == "IN_QUEUE":
            if queue_started_at is None:
                queue_started_at = now()
            time_in_queue = now() - queue_started_at
            if time_in_queue >= queue_stall_s:
                classification = CAPACITY_UNAVAILABLE if is_retry_attempt else WORKER_PROVISIONING_STALLED
                log(
                    OrchestrationEvent(
                        "health_queue_stalled",
                        now(),
                        {"job_id": job_id, "time_in_queue_s": time_in_queue, "classification": classification},
                    )
                )
                return HealthOutcome(
                    classification=classification,
                    passed=False,
                    job_id=job_id,
                    time_in_queue_s=time_in_queue,
                    final_status=status,
                )
            # still inside the grace/stall window (whether under or over
            # queue_grace_s -- grace only affects log verbosity, never the
            # fail decision itself). Keep waiting.

        sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# Endpoint teardown helper, reused between bounded-retry attempts (item 3)
# ---------------------------------------------------------------------------
def cancel_job_if_active(
    transport: Transport, endpoint_id: str, api_key: str, job_id: Optional[str], *, log: LogFn = _default_log
) -> None:
    if not job_id:
        return
    headers = {"Authorization": f"Bearer {api_key}"}
    status_resp = transport.request(
        "GET", f"{RUNPOD_SERVERLESS_BASE}/{endpoint_id}/status/{job_id}", headers=headers
    )
    status = (status_resp.json_body or {}).get("status") if status_resp.status_code == 200 else None
    if status in {"IN_QUEUE", "IN_PROGRESS"}:
        transport.request(
            "POST", f"{RUNPOD_SERVERLESS_BASE}/{endpoint_id}/cancel/{job_id}", headers=headers
        )
        log(OrchestrationEvent("job_cancelled", time.monotonic(), {"job_id": job_id, "prior_status": status}))


# ---------------------------------------------------------------------------
# Bounded infra retry wrapper (D-041 item 3)
# ---------------------------------------------------------------------------
@dataclass
class AttemptResult:
    readiness: EndpointReadiness
    health: Optional[HealthOutcome]


@dataclass
class OrchestrationResult:
    passed: bool
    classification: str
    attempts: list[AttemptResult]
    total_elapsed_s: float


def run_with_bounded_retry(
    roll_fn: Callable[[], None],
    attempt_fn: Callable[[bool], AttemptResult],
    *,
    max_infra_retries: int = 1,
    backoff_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    log: LogFn = _default_log,
) -> OrchestrationResult:
    """Runs `attempt_fn` (readiness + health, one full attempt) up to
    `1 + max_infra_retries` times total. Only infra-class failures
    (INFRA_FAILURE_CLASSES) are retried; HEALTH_APP_FAILURE stops immediately
    with no retry. `roll_fn` re-rolls the endpoint before each retry attempt
    (never before the first -- that roll already happened by the time this
    is called). Bounded: `max_infra_retries` is a hard ceiling, never a loop.
    """
    start = now()
    attempts: list[AttemptResult] = []
    for attempt_index in range(max_infra_retries + 1):
        is_retry = attempt_index > 0
        if is_retry:
            log(OrchestrationEvent("infra_retry_backoff", now(), {"backoff_s": backoff_s, "attempt": attempt_index}))
            sleep(backoff_s)
            roll_fn()
        result = attempt_fn(is_retry)
        attempts.append(result)

        if result.readiness.ready and result.health is not None and result.health.passed:
            return OrchestrationResult(
                passed=True, classification=HEALTH_PASSED, attempts=attempts, total_elapsed_s=now() - start
            )

        classification = (
            result.readiness.classification
            if not result.readiness.ready
            else (result.health.classification if result.health is not None else RUNPOD_API_ERROR)
        )
        if classification not in INFRA_FAILURE_CLASSES:
            # HEALTH_APP_FAILURE (or an unrecognized classification): never
            # retried here -- a real application/CUDA problem is not a flake.
            return OrchestrationResult(
                passed=False, classification=classification, attempts=attempts, total_elapsed_s=now() - start
            )
        # infra-class failure and retries remain -> loop retries; if this was
        # the last allowed attempt, fall through and report it below.

    return OrchestrationResult(
        passed=False, classification=classification, attempts=attempts, total_elapsed_s=now() - start
    )


# ---------------------------------------------------------------------------
# CLI entry point used by the workflow. Not exercised by unit tests (those
# call the functions above directly with fakes); this only wires real env
# vars/GITHUB_ENV/GITHUB_OUTPUT to the real RunPod REST/serverless APIs.
# ---------------------------------------------------------------------------
def _write_github_env(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_ENV")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


def main() -> int:
    endpoint_id = os.environ["ENDPOINT_ID"]
    api_key = os.environ["RUNPOD_API_KEY"]
    template_id = os.environ["TEMPLATE_ID"]
    workers_max = int(os.environ.get("ENDPOINT_WORKERS_MAX", "1"))

    transport = UrllibTransport()

    def roll() -> None:
        _default_log(OrchestrationEvent("endpoint_roll_started_at", time.time(), {"template_id": template_id}))
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "templateId": template_id,
            "workersMin": 0,
            "workersMax": workers_max,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 2,
            "idleTimeout": 5,
            "executionTimeoutMs": 1800000,
        }
        transport.request(
            "PATCH", f"{RUNPOD_REST_BASE}/endpoints/{endpoint_id}", headers=headers, json_body=payload
        )

    roll()

    def attempt(is_retry: bool) -> AttemptResult:
        readiness = wait_for_endpoint_ready(transport, endpoint_id, api_key, template_id, workers_max)
        if not readiness.ready:
            return AttemptResult(readiness=readiness, health=None)
        health = submit_and_poll_health(transport, endpoint_id, api_key, is_retry_attempt=is_retry)
        if not health.passed:
            cancel_job_if_active(transport, endpoint_id, api_key, health.job_id)
        return AttemptResult(readiness=readiness, health=health)

    result = run_with_bounded_retry(roll, attempt)

    last_health_job_id = next(
        (a.health.job_id for a in reversed(result.attempts) if a.health is not None and a.health.job_id),
        "",
    )
    _write_github_env("HEALTH_JOB_ID", last_health_job_id)
    _write_github_env("RUNPOD_INFRA_CLASSIFICATION", result.classification)

    print(f"[runpod_orchestration] final classification={result.classification} passed={result.passed} "
          f"attempts={len(result.attempts)} total_elapsed_s={result.total_elapsed_s:.1f}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
