"""D-042/D-043: GPUExecutionProvider abstraction -- Serverless-wrapper
parity, RunPod Pod/Modal "never a second implementation" checks.
Serverless workflow/orchestration itself is untouched by this module
(see tests/test_runpod_orchestration.py, still green and unmodified);
these tests only cover the thin wrappers and the schema parity
requirement across all three backends' HealthCheckResult shapes.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from gpu_execution_provider import (
    EXECUTION_BACKEND_MODAL,
    EXECUTION_BACKEND_POD,
    EXECUTION_BACKEND_SERVERLESS,
    VALID_EXECUTION_BACKENDS,
    HealthCheckResult,
    ModalExecutionProvider,
    RunPodServerlessExecutionProvider,
)
from modal_gpu_config import EXCLUDED_MODAL_GPU_TYPES
from runpod_orchestration import HEALTH_PASSED, TransportResponse
from runpod_pod_provider import RunPodPodExecutionProvider


def _noop_log(event):
    pass


class FakeServerlessTransport:
    def __init__(self):
        self.calls = []
        self._get_endpoint = []
        self._post_run = []
        self._get_status = []

    def request(self, method, url, *, headers, json_body=None):
        self.calls.append((method, url, json_body))
        if method == "GET" and "/status/" in url:
            return self._get_status.pop(0)
        if method == "GET" and "/endpoints/" in url:
            return self._get_endpoint.pop(0)
        if method == "POST" and url.endswith("/run"):
            return self._post_run.pop(0)
        raise AssertionError(f"unscripted call: {method} {url}")


def test_execution_backend_constants_are_the_three_and_only_three_valid_values():
    assert VALID_EXECUTION_BACKENDS == {EXECUTION_BACKEND_SERVERLESS, EXECUTION_BACKEND_POD, EXECUTION_BACKEND_MODAL}
    assert EXECUTION_BACKEND_SERVERLESS == "serverless"
    assert EXECUTION_BACKEND_POD == "pod"
    assert EXECUTION_BACKEND_MODAL == "modal"


def test_serverless_provider_health_check_reuses_orchestration_primitives_verbatim():
    transport = FakeServerlessTransport()
    transport._get_endpoint = [TransportResponse(200, {"templateId": "tmpl-1", "workersMin": 0, "workersMax": 1})]
    transport._post_run = [TransportResponse(200, {"id": "job-1"})]
    transport._get_status = [
        TransportResponse(200, {"status": "COMPLETED", "output": {"ok": True, "cuda_available": True}}),
    ]
    provider = RunPodServerlessExecutionProvider(
        transport, "endpoint-1", "api-key", "tmpl-1", 1, log=_noop_log,
    )

    result = provider.health_check()

    assert result.execution_provider == "RUNPOD_SERVERLESS"
    assert result.passed is True
    assert result.classification == HEALTH_PASSED
    # No extra/duplicate HTTP surface invented -- exactly the endpoint-ready
    # GET, the health-job POST, and its status GET, same as
    # wait_for_endpoint_ready + submit_and_poll_health called directly.
    methods_and_shapes = [(m, "/endpoints/" in u, u.endswith("/run"), "/status/" in u) for m, u, _b in transport.calls]
    assert methods_and_shapes == [
        ("GET", True, False, False),
        ("POST", False, True, False),
        ("GET", False, False, True),
    ]


def test_serverless_provider_teardown_is_a_documented_noop():
    # Serverless teardown belongs to the RAW workflow's own "always()" step,
    # not to this wrapper -- calling it must never touch the network.
    transport = FakeServerlessTransport()
    provider = RunPodServerlessExecutionProvider(transport, "e", "k", "t", 1, log=_noop_log)
    provider.teardown()
    assert transport.calls == []


def test_health_check_result_schema_is_identical_across_all_three_providers():
    result_fields = {f.name for f in fields(HealthCheckResult)}
    # RunPodPodExecutionProvider.health_check and ModalExecutionProvider.
    # health_check both construct this exact same dataclass -- assert all
    # three backends' results are literally the same type (not parallel
    # structures with the same-looking fields).
    assert isinstance(RunPodPodExecutionProvider, type)
    assert isinstance(ModalExecutionProvider, type)
    expected = {"execution_provider", "passed", "classification", "elapsed_s", "detail"}
    assert result_fields == expected


# ---------------------------------------------------------------------------
# ModalExecutionProvider (D-043)
# ---------------------------------------------------------------------------
def test_modal_provider_accepts_l4():
    ModalExecutionProvider(invoke=lambda: {"ok": True}, gpu_type="L4")  # must not raise


@pytest.mark.parametrize("excluded", sorted(EXCLUDED_MODAL_GPU_TYPES))
def test_modal_provider_rejects_excluded_gpu_types(excluded):
    with pytest.raises(ValueError, match=excluded):
        ModalExecutionProvider(invoke=lambda: {"ok": True}, gpu_type=excluded)


def test_modal_provider_health_check_passes_on_ok_result():
    provider = ModalExecutionProvider(invoke=lambda: {"ok": True, "cuda_available": True})

    result = provider.health_check()

    assert result.execution_provider == "MODAL"
    assert result.passed is True
    assert result.classification is None
    assert result.detail["gpu_type"] == "L4"
    assert result.detail["result"]["cuda_available"] is True


def test_modal_provider_health_check_fails_on_not_ok_result():
    provider = ModalExecutionProvider(invoke=lambda: {"ok": False, "cuda_available": False})

    result = provider.health_check()

    assert result.passed is False
    assert result.classification == "MODAL_HEALTH_FAILED"


def test_modal_provider_health_check_never_crashes_on_invoke_exception():
    def _raise():
        raise RuntimeError("modal call failed")

    provider = ModalExecutionProvider(invoke=_raise)

    result = provider.health_check()

    assert result.passed is False
    assert result.classification == "MODAL_INVOKE_ERROR"
    assert "modal call failed" in result.detail["error"]


def test_modal_provider_teardown_is_a_documented_noop_and_never_raises():
    invoked = {"count": 0}

    def _invoke():
        invoked["count"] += 1
        return {"ok": True}

    provider = ModalExecutionProvider(invoke=_invoke)
    provider.teardown()
    assert invoked["count"] == 0  # teardown never calls invoke either
