"""GPUExecutionProvider abstraction (D-042: CutSell QA GPU execution
fallback -- RunPod Pod On-Demand automation; D-043: Modal GPU execution --
first live validation. KEEP SERVERLESS AND RUNPOD POD FULLY AVAILABLE;
Modal is an ADDITIONAL backend, not a replacement).

CutSell supports three interchangeable GPU execution backends --
`RUNPOD_SERVERLESS`, `RUNPOD_POD`, and `MODAL` -- behind this one
interface. The editorial engine (`cutsell_worker.serverless_handler.run_op`)
does not know or care which backend executed it: every provider ultimately
invokes that exact same canonical dispatcher with the exact same payload
shapes. This module only decides HOW a job reaches that dispatcher
(RunPod's Serverless job-queue envelope, a direct HTTP call to a running
Pod, or a Modal Function call), never WHAT runs.

`RunPodServerlessExecutionProvider` below is a thin wrapper around the
already-tested D-041 `runpod_orchestration.py` primitives -- it does not
duplicate `wait_for_endpoint_ready`/`submit_and_poll_health`, it calls them
directly. See `runpod_pod_provider.py` for `RunPodPodExecutionProvider`,
the RunPod Pod on-demand implementation, and `ModalExecutionProvider`
below for the Modal backend (D-043 first phase: health/smoke-check only,
same as the other two -- `run_benchmark`/Video00-on-Modal integration is
separately gated, not part of this phase's interface).

Serverless and RunPod Pod remain fully available and untouched by this
module (no import here can execute code that would touch a real
endpoint/Pod); this only adds a shared reporting shape
(`HealthCheckResult`) and a selector constant trio so a caller can pick a
backend by name (`execution_backend: "serverless" | "pod" | "modal"`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from modal_gpu_config import require_modal_gpu_type
from runpod_orchestration import (
    HealthOutcome,
    LogFn,
    Transport,
    _default_log,
    submit_and_poll_health,
    wait_for_endpoint_ready,
)

EXECUTION_BACKEND_SERVERLESS = "serverless"
EXECUTION_BACKEND_POD = "pod"
EXECUTION_BACKEND_MODAL = "modal"
VALID_EXECUTION_BACKENDS = frozenset({EXECUTION_BACKEND_SERVERLESS, EXECUTION_BACKEND_POD, EXECUTION_BACKEND_MODAL})


@dataclass(frozen=True)
class HealthCheckResult:
    """Backend-agnostic health-check outcome. Both providers return this
    same shape (output-format parity, D-042) so downstream reporting never
    needs to special-case which backend ran. `execution_provider` is the
    literal string that belongs in a result's `execution_provider` field
    (`RUNPOD_SERVERLESS` or `RUNPOD_POD`)."""

    execution_provider: str
    passed: bool
    classification: Optional[str]
    elapsed_s: float
    detail: dict = field(default_factory=dict)


class GPUExecutionProvider(Protocol):
    """One canonical benchmark contract (D-042): every implementation
    resolves/prepares its own GPU execution target, runs a health check
    against it, and guarantees cleanup -- but never runs a different
    editorial pipeline. Full-benchmark execution (`run_benchmark`) is
    deliberately not part of this first phase's interface; the standing
    gate is health-check-only until the user authorizes the first full
    Pod benchmark."""

    def health_check(self) -> HealthCheckResult: ...

    def teardown(self) -> None: ...


class RunPodServerlessExecutionProvider:
    """Existing production backend, wrapped behind `GPUExecutionProvider`
    for parity with the new Pod backend -- not a reimplementation. Reuses
    `wait_for_endpoint_ready` + `submit_and_poll_health` verbatim."""

    def __init__(
        self,
        transport: Transport,
        endpoint_id: str,
        api_key: str,
        expected_template_id: str,
        expected_workers_max: int,
        *,
        log: LogFn = _default_log,
    ) -> None:
        self._transport = transport
        self._endpoint_id = endpoint_id
        self._api_key = api_key
        self._expected_template_id = expected_template_id
        self._expected_workers_max = expected_workers_max
        self._log = log

    def health_check(self) -> HealthCheckResult:
        readiness = wait_for_endpoint_ready(
            self._transport,
            self._endpoint_id,
            self._api_key,
            self._expected_template_id,
            self._expected_workers_max,
            log=self._log,
        )
        if not readiness.ready:
            return HealthCheckResult(
                execution_provider="RUNPOD_SERVERLESS",
                passed=False,
                classification=readiness.classification,
                elapsed_s=readiness.elapsed_s,
                detail={"stage": "endpoint_readiness", "endpoint_id": self._endpoint_id},
            )
        outcome: HealthOutcome = submit_and_poll_health(
            self._transport,
            self._endpoint_id,
            self._api_key,
            log=self._log,
        )
        return HealthCheckResult(
            execution_provider="RUNPOD_SERVERLESS",
            passed=bool(outcome.passed),
            classification=outcome.classification,
            elapsed_s=outcome.time_in_queue_s,
            detail={
                "stage": "health_job",
                "endpoint_id": self._endpoint_id,
                "job_id": outcome.job_id,
                "final_status": outcome.final_status,
                "raw_detail": outcome.detail,
            },
        )

    def teardown(self) -> None:
        # Serverless teardown (workersMax=0 PATCH, template delete) is
        # already owned by the existing RAW workflow's own "Teardown RunPod
        # benchmark resources" `if: always()` step -- deliberately a no-op
        # here rather than a second, divergent teardown path for the same
        # resource.
        return None


class ModalExecutionProvider:
    """D-043: third `GPUExecutionProvider` implementation, for Modal's
    serverless GPU backend. Health/smoke-check only in this phase, exactly
    like the other two providers -- `run_benchmark`/Video00-on-Modal
    integration is separately gated, not part of this interface yet.

    Deliberately does not hardcode a Modal SDK call (`Function.lookup` vs
    `Function.from_name` etc. have changed across Modal SDK versions, and
    this repo does not pin one yet) -- `invoke` is an injected zero-arg
    callable returning a dict with at least an `ok` key (the exact shape
    `modal_gpu_diagnostics.collect_gpu_diagnostics()` already returns).
    This keeps the class fully unit-testable without the `modal` package
    installed and keeps the actual SDK-call-shape decision for whenever
    Video00-on-Modal integration is authorized, rather than guessing it
    now.

    Unlike the two RunPod backends, Modal's serverless GPU functions scale
    to zero automatically once the call returns -- there is no persistent
    Pod/endpoint for `teardown()` to stop or delete."""

    def __init__(self, invoke: Callable[[], dict], *, gpu_type: str = "L4") -> None:
        require_modal_gpu_type(gpu_type)
        self._invoke = invoke
        self._gpu_type = gpu_type

    def health_check(self) -> HealthCheckResult:
        start = time.monotonic()
        try:
            result = self._invoke()
        except Exception as exc:  # noqa: BLE001 -- an invoke failure IS the health result, never an unhandled crash
            return HealthCheckResult(
                execution_provider="MODAL",
                passed=False,
                classification="MODAL_INVOKE_ERROR",
                elapsed_s=time.monotonic() - start,
                detail={"gpu_type": self._gpu_type, "error": str(exc)},
            )
        passed = bool(result.get("ok"))
        return HealthCheckResult(
            execution_provider="MODAL",
            passed=passed,
            classification=None if passed else "MODAL_HEALTH_FAILED",
            elapsed_s=time.monotonic() - start,
            detail={"gpu_type": self._gpu_type, "result": result},
        )

    def teardown(self) -> None:
        # Modal's serverless GPU functions scale to zero automatically --
        # there is no persistent Pod/endpoint for this method to stop or
        # delete, unlike the two RunPod backends. Documented no-op, not a
        # missing feature.
        return None
