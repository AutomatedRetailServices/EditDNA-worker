"""D-042: RunPod Pod On-Demand execution provider -- mock/test scenarios
required before implementation touches any real GPU. Fully scripted fake
Transport + fake clock, mirroring tests/test_runpod_orchestration.py's own
pattern -- no network, no RunPod credentials, no paid GPU.
"""
from __future__ import annotations

from runpod_orchestration import TransportResponse
from runpod_pod_provider import (
    APPROVED_POD_GPU_TYPE_IDS,
    EXCLUDED_POD_GPU_TYPE_IDS,
    POD_CAPACITY_UNAVAILABLE,
    POD_COST_CEILING_EXCEEDED,
    POD_CREATED_FRESH,
    POD_HEALTH_APP_FAILURE,
    POD_HEALTH_PASSED,
    POD_REUSED,
    POD_RESTARTED,
    POD_RESTART_UNAVAILABLE,
    POD_RUNPOD_API_ERROR,
    POD_STALE_RECREATED,
    GPUCandidate,
    PodExecutionConfig,
    RunPodPodExecutionProvider,
    create_pod,
    fetch_pod_gpu_catalog,
    rank_gpu_candidates,
)


def _noop_log(event):
    pass


class FakePodTransport:
    """Scripted fake covering the RunPod Pod REST surface this module
    calls: GET/POST/DELETE /v1/pods[...], GET /v1/gpuTypes. Each queue is
    popped in call order; an empty/mismatched queue raises loudly."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.get_pod: list[TransportResponse] = []
        self.post_pods: list[TransportResponse] = []
        self.post_start: list[TransportResponse] = []
        self.post_stop: list[TransportResponse] = []
        self.delete_pod: list[TransportResponse] = []
        self.get_gpu_types: list[TransportResponse] = []

    def request(self, method, url, *, headers, json_body=None):
        self.calls.append((method, url, json_body))
        if method == "GET" and ("/gpuTypes" in url or "/gpu-types" in url):
            return self._pop(self.get_gpu_types, f"GET gpuTypes ({url})")
        if method == "GET" and "/pods/" in url:
            return self._pop(self.get_pod, f"GET pod ({url})")
        if method == "POST" and url.endswith("/pods"):
            return self._pop(self.post_pods, f"POST pods ({url})")
        if method == "POST" and url.endswith("/start"):
            return self._pop(self.post_start, f"POST start ({url})")
        if method == "POST" and url.endswith("/stop"):
            return self._pop(self.post_stop, f"POST stop ({url})")
        if method == "DELETE" and "/pods/" in url:
            return self._pop(self.delete_pod, f"DELETE pod ({url})")
        raise AssertionError(f"unscripted call: {method} {url}")

    @staticmethod
    def _pop(queue, label):
        if not queue:
            raise AssertionError(f"no more scripted responses for {label}")
        return queue.pop(0)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _config(**overrides) -> PodExecutionConfig:
    base = dict(
        api_key="fake-key",
        image="ghcr.io/example/cutsell-serverless@sha256:deadbeef",
        pod_name="cutsell-qa-pod",
        cost_ceiling_usd_per_hr=1.50,
        restart_wait_timeout_s=30.0,
        poll_interval_s=5.0,
    )
    base.update(overrides)
    return PodExecutionConfig(**base)


def _provider(transport, existing_pod_id=None, http_get=None, **cfg_overrides):
    clock = FakeClock()
    return RunPodPodExecutionProvider(
        transport,
        _config(**cfg_overrides),
        existing_pod_id=existing_pod_id,
        http_get=http_get,
        now=clock.now,
        sleep=clock.sleep,
        log=_noop_log,
    ), clock


# ---------------------------------------------------------------------------
# 1. Stopped Pod successfully reused (restart)
# ---------------------------------------------------------------------------
def test_stopped_pod_successfully_reused_via_restart():
    transport = FakePodTransport()
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-1", "status": "EXITED", "imageName": "ghcr.io/example/cutsell-serverless@sha256:deadbeef"}),
        TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "ghcr.io/example/cutsell-serverless@sha256:deadbeef"}),
    ]
    transport.post_start = [TransportResponse(200, {})]
    provider, _clock = _provider(transport, existing_pod_id="pod-1")

    result = provider.ensure_ready()

    assert result.classification == POD_RESTARTED
    assert result.pod_id == "pod-1"


# ---------------------------------------------------------------------------
# 2. Restart succeeds after more than one poll
# ---------------------------------------------------------------------------
def test_restart_succeeds_after_polling():
    transport = FakePodTransport()
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-1", "status": "EXITED", "imageName": "img"}),
        TransportResponse(200, {"id": "pod-1", "status": "PENDING", "imageName": "img"}),
        TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"}),
    ]
    transport.post_start = [TransportResponse(200, {})]
    provider, _clock = _provider(transport, existing_pod_id="pod-1", image="img")

    result = provider.ensure_ready()

    assert result.classification == POD_RESTARTED
    assert result.pod_id == "pod-1"


# ---------------------------------------------------------------------------
# 3. Restart unavailable -> falls through to stale-recreate path
# ---------------------------------------------------------------------------
def test_restart_unavailable_falls_through_to_recreate():
    transport = FakePodTransport()
    still_exited = TransportResponse(200, {"id": "pod-1", "status": "EXITED", "imageName": "img"})
    transport.get_pod = [
        still_exited,  # initial inspect
        still_exited,  # wait_for_pod_running poll @ t=0
        still_exited,  # wait_for_pod_running poll @ t=5
        still_exited,  # wait_for_pod_running poll @ t=10 -> elapsed>=timeout, give up
    ]
    transport.post_start = [TransportResponse(200, {})]
    transport.delete_pod = [TransportResponse(204, None)]
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-2"})]
    provider, clock = _provider(transport, existing_pod_id="pod-1", image="img", restart_wait_timeout_s=10.0, poll_interval_s=5.0)

    result = provider.ensure_ready()

    assert result.classification == POD_STALE_RECREATED
    assert result.pod_id == "pod-2"
    assert ("DELETE", "https://rest.runpod.io/v1/pods/pod-1", None) in transport.calls


# ---------------------------------------------------------------------------
# 4. Stale Pod (RUNNING but wrong image) deleted, fresh one created
# ---------------------------------------------------------------------------
def test_stale_pod_wrong_image_deleted_and_recreated():
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "old-image"})]
    transport.delete_pod = [TransportResponse(204, None)]
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-2"})]
    provider, _clock = _provider(transport, existing_pod_id="pod-1", image="new-image")

    result = provider.ensure_ready()

    assert result.classification == POD_STALE_RECREATED
    assert result.pod_id == "pod-2"


# ---------------------------------------------------------------------------
# 5. No existing Pod -> fresh Pod created
# ---------------------------------------------------------------------------
def test_fresh_pod_created_when_none_exists():
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-new"})]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.classification == POD_CREATED_FRESH
    assert result.pod_id == "pod-new"
    assert result.gpu_selection.chosen.gpu_type_id == APPROVED_POD_GPU_TYPE_IDS[0]


# ---------------------------------------------------------------------------
# 6-8. GPU fallback chain: 4090 unavailable -> A40 unavailable -> A6000 chosen
# ---------------------------------------------------------------------------
def test_gpu_fallback_chain_rtx4090_then_a40_then_a6000():
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [
        TransportResponse(400, {"error": "no instances currently available"}),  # RTX 4090
        TransportResponse(400, {"error": "insufficient capacity"}),  # A40
        TransportResponse(201, {"id": "pod-a6000"}),  # RTX A6000
    ]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.classification == POD_CREATED_FRESH
    assert result.pod_id == "pod-a6000"
    assert result.gpu_selection.chosen.gpu_type_id == "NVIDIA RTX A6000"
    attempted_gpu_ids = [c["gpuTypeIds"][0] for _m, _u, c in transport.calls if _m == "POST" and _u.endswith("/pods")]
    assert attempted_gpu_ids == ["NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA RTX A6000"]


def test_gpu_fallback_chain_reaches_l4_last():
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [
        TransportResponse(400, {"error": "not available"}),
        TransportResponse(400, {"error": "not available"}),
        TransportResponse(400, {"error": "not available"}),
        TransportResponse(201, {"id": "pod-l4"}),
    ]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.pod_id == "pod-l4"
    assert result.gpu_selection.chosen.gpu_type_id == "NVIDIA L4"


# ---------------------------------------------------------------------------
# 9. No compatible GPU available -> CAPACITY_UNAVAILABLE, nothing provisioned
# ---------------------------------------------------------------------------
def test_no_compatible_gpu_available_reports_capacity_unavailable():
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [
        TransportResponse(400, {"error": "no instances available"}),
        TransportResponse(400, {"error": "no instances available"}),
        TransportResponse(400, {"error": "no instances available"}),
        TransportResponse(400, {"error": "no instances available"}),
    ]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.classification == POD_CAPACITY_UNAVAILABLE
    assert result.pod_id is None


# ---------------------------------------------------------------------------
# 10. Expensive/unapproved GPU rejected by the cost ceiling before any
#     creation attempt is even made for it
# ---------------------------------------------------------------------------
def test_gpu_priced_over_cost_ceiling_is_never_attempted():
    catalog = [
        {"id": "NVIDIA GeForce RTX 4090", "communityPrice": 5.00, "communityCloud": True},  # priced absurdly, over ceiling
        {"id": "NVIDIA A40", "communityPrice": 0.40, "communityCloud": True},
        {"id": "NVIDIA RTX A6000", "communityPrice": 0.50, "communityCloud": True},
        {"id": "NVIDIA L4", "communityPrice": 0.30, "communityCloud": True},
    ]
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, catalog)]
    transport.post_pods = [TransportResponse(201, {"id": "pod-a40"})]
    provider, _clock = _provider(transport, existing_pod_id=None, cost_ceiling_usd_per_hr=1.50)

    result = provider.ensure_ready()

    assert result.pod_id == "pod-a40"
    assert result.gpu_selection.chosen.gpu_type_id == "NVIDIA A40"
    attempted_gpu_ids = [c["gpuTypeIds"][0] for _m, _u, c in transport.calls if _m == "POST" and _u.endswith("/pods")]
    assert "NVIDIA GeForce RTX 4090" not in attempted_gpu_ids


def test_all_approved_gpus_over_ceiling_reports_cost_ceiling_exceeded():
    catalog = [{"id": g, "communityPrice": 9.99, "communityCloud": True} for g in APPROVED_POD_GPU_TYPE_IDS]
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, catalog)]
    provider, _clock = _provider(transport, existing_pod_id=None, cost_ceiling_usd_per_hr=1.50)

    result = provider.ensure_ready()

    assert result.classification == POD_COST_CEILING_EXCEEDED
    assert result.pod_id is None
    assert not any(m == "POST" and u.endswith("/pods") for m, u, _b in transport.calls)


# ---------------------------------------------------------------------------
# 11. Blackwell (and other excluded GPUs) never appear in the approved pool
#     or get attempted, no matter what the catalog reports
# ---------------------------------------------------------------------------
def test_blackwell_never_in_approved_pool_or_attempted():
    assert "NVIDIA RTX PRO 6000 Blackwell Server Edition" in EXCLUDED_POD_GPU_TYPE_IDS
    assert "NVIDIA RTX PRO 6000 Blackwell Server Edition" not in APPROVED_POD_GPU_TYPE_IDS

    catalog = [
        {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "communityPrice": 0.01, "communityCloud": True},
    ]
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, catalog)]
    transport.post_pods = [TransportResponse(201, {"id": "pod-4090"})]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    # Blackwell was the cheapest thing in the catalog but must never be
    # attempted -- the RTX 4090 (absent from the catalog, price unknown)
    # is still preferred because it's approved.
    assert result.gpu_selection.chosen.gpu_type_id == "NVIDIA GeForce RTX 4090"
    attempted_gpu_ids = [c["gpuTypeIds"][0] for _m, _u, c in transport.calls if _m == "POST" and _u.endswith("/pods")]
    assert "NVIDIA RTX PRO 6000 Blackwell Server Edition" not in attempted_gpu_ids


# ---------------------------------------------------------------------------
# 12. Health failure is reported, never silently treated as pass
# ---------------------------------------------------------------------------
def test_health_failure_reported_as_pod_health_app_failure():
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"})]
    provider, _clock = _provider(
        transport,
        existing_pod_id="pod-1",
        image="img",
        http_get=lambda url: (200, {"ok": True, "cuda_available": False}),
    )

    result = provider.health_check()

    assert result.execution_provider == "RUNPOD_POD"
    assert result.passed is False
    assert result.classification == POD_HEALTH_APP_FAILURE


def test_health_pass_reported_correctly():
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"})]
    provider, _clock = _provider(
        transport,
        existing_pod_id="pod-1",
        image="img",
        http_get=lambda url: (200, {"ok": True, "cuda_available": True, "device_name": "NVIDIA A40"}),
    )

    result = provider.health_check()

    assert result.passed is True
    assert result.classification == POD_HEALTH_PASSED
    assert result.detail["health_payload"]["device_name"] == "NVIDIA A40"


def test_health_check_never_reaches_video00_stage_when_lifecycle_fails():
    # ensure_ready() itself failing (no pod_id) must short-circuit
    # health_check before any HTTP GET to the pod is attempted.
    calls = []
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(400, {"error": "no instances available"})] * 4
    provider, _clock = _provider(
        transport,
        existing_pod_id=None,
        http_get=lambda url: calls.append(url) or (200, {"ok": True, "cuda_available": True}),
    )

    result = provider.health_check()

    assert result.passed is False
    assert result.classification == POD_CAPACITY_UNAVAILABLE
    assert calls == []  # never attempted a health GET against a Pod that doesn't exist


# ---------------------------------------------------------------------------
# 13. One recreation fallback succeeds
# ---------------------------------------------------------------------------
def test_one_recreation_fallback_succeeds_after_restart_failure():
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "ERROR", "imageName": "img"})]
    transport.delete_pod = [TransportResponse(204, None)]
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-2"})]
    provider, _clock = _provider(transport, existing_pod_id="pod-1", image="img")

    result = provider.ensure_ready()

    assert result.classification == POD_STALE_RECREATED
    assert result.pod_id == "pod-2"


# ---------------------------------------------------------------------------
# 14. Recreation is bounded -- exactly one fresh-creation cycle is attempted,
#     never an unbounded restart/recreate loop
# ---------------------------------------------------------------------------
def test_recreation_is_bounded_not_looped():
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "ERROR", "imageName": "img"})]
    transport.delete_pod = [TransportResponse(204, None)]
    transport.get_gpu_types = [TransportResponse(200, [])]
    # Even if every single GPU creation attempt fails, ensure_ready must
    # return (CAPACITY_UNAVAILABLE) rather than looping back to re-inspect
    # pod-1 again.
    transport.post_pods = [TransportResponse(400, {"error": "no instances available"})] * len(APPROVED_POD_GPU_TYPE_IDS)
    provider, _clock = _provider(transport, existing_pod_id="pod-1", image="img")

    result = provider.ensure_ready()

    assert result.classification == POD_CAPACITY_UNAVAILABLE
    get_pod_calls = [c for c in transport.calls if c[0] == "GET" and "/pods/" in c[1]]
    assert len(get_pod_calls) == 1  # pod-1 inspected exactly once, never re-polled in a loop


# ---------------------------------------------------------------------------
# 15. STOP always runs after a successful health check
# ---------------------------------------------------------------------------
def test_stop_runs_after_health_check_success():
    transport = FakePodTransport()
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"}),
        TransportResponse(200, {"id": "pod-1", "status": "EXITED", "imageName": "img"}),
    ]
    transport.post_stop = [TransportResponse(200, {})]
    provider, _clock = _provider(
        transport, existing_pod_id="pod-1", image="img",
        http_get=lambda url: (200, {"ok": True, "cuda_available": True}),
    )

    try:
        result = provider.health_check()
        assert result.passed is True
    finally:
        provider.teardown()

    assert ("POST", "https://rest.runpod.io/v1/pods/pod-1/stop", {}) in transport.calls


# ---------------------------------------------------------------------------
# 16. STOP always runs after a failed health check
# ---------------------------------------------------------------------------
def test_stop_runs_after_health_check_failure():
    transport = FakePodTransport()
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"}),
        TransportResponse(200, {"id": "pod-1", "status": "EXITED", "imageName": "img"}),
    ]
    transport.post_stop = [TransportResponse(200, {})]
    provider, _clock = _provider(
        transport, existing_pod_id="pod-1", image="img",
        http_get=lambda url: (200, {"ok": True, "cuda_available": False}),
    )

    try:
        result = provider.health_check()
        assert result.passed is False
    finally:
        provider.teardown()

    assert any(m == "POST" and u.endswith("/stop") for m, u, _b in transport.calls)


# ---------------------------------------------------------------------------
# 17. STOP always runs even when the caller's own code raises
# ---------------------------------------------------------------------------
def test_stop_runs_after_caller_exception_via_finally():
    transport = FakePodTransport()
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"}),
        TransportResponse(200, {"id": "pod-1", "status": "EXITED", "imageName": "img"}),
    ]
    transport.post_stop = [TransportResponse(200, {})]
    provider, _clock = _provider(
        transport, existing_pod_id="pod-1", image="img",
        http_get=lambda url: (200, {"ok": True, "cuda_available": True}),
    )

    raised = False
    try:
        provider.health_check()
        raise RuntimeError("simulated downstream failure after health passed")
    except RuntimeError:
        raised = True
    finally:
        provider.teardown()

    assert raised is True
    assert any(m == "POST" and u.endswith("/stop") for m, u, _b in transport.calls)


# ---------------------------------------------------------------------------
# 18. STOP API failure is retried once
# ---------------------------------------------------------------------------
def test_stop_api_failure_is_retried_once():
    transport = FakePodTransport()
    transport.post_stop = [TransportResponse(500, {}), TransportResponse(200, {})]
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "EXITED", "imageName": "img"})]
    provider, _clock = _provider(transport, existing_pod_id="pod-1", image="img")
    provider._pod_id = "pod-1"  # simulate a Pod already known to this instance

    provider.teardown()

    stop_calls = [c for c in transport.calls if c[0] == "POST" and c[1].endswith("/stop")]
    assert len(stop_calls) == 2


def test_teardown_is_a_noop_when_no_pod_was_ever_acquired():
    transport = FakePodTransport()
    provider, _clock = _provider(transport, existing_pod_id=None)

    provider.teardown()  # must not raise, must not call the API at all

    assert transport.calls == []


# ---------------------------------------------------------------------------
# 19. GPU catalog cross-check tested directly (availability search building
#     block used by rank_gpu_candidates/ensure_ready above)
# ---------------------------------------------------------------------------
def test_fetch_pod_gpu_catalog_parses_price_and_availability():
    transport = FakePodTransport()
    transport.get_gpu_types = [
        TransportResponse(200, [
            {"id": "NVIDIA A40", "communityPrice": 0.39, "communityCloud": True},
            {"id": "NVIDIA L4", "securePrice": 0.29, "secureCloud": False},
        ])
    ]
    catalog = fetch_pod_gpu_catalog(transport, "fake-key", log=_noop_log)
    assert catalog["NVIDIA A40"] == {"price_usd_per_hr": 0.39, "available": True}
    assert catalog["NVIDIA L4"] == {"price_usd_per_hr": 0.29, "available": False}


def test_fetch_pod_gpu_catalog_never_raises_on_unavailable_catalog():
    transport = FakePodTransport()
    # Both the primary and fallback catalog URLs are tried before giving up.
    transport.get_gpu_types = [TransportResponse(500, None), TransportResponse(500, None)]
    catalog = fetch_pod_gpu_catalog(transport, "fake-key", log=_noop_log)
    assert catalog == {}


def test_rank_gpu_candidates_skips_explicitly_unavailable_catalog_entries():
    catalog = {
        "NVIDIA GeForce RTX 4090": {"price_usd_per_hr": 0.50, "available": False},
        "NVIDIA A40": {"price_usd_per_hr": 0.40, "available": True},
    }
    selection = rank_gpu_candidates(catalog, cost_ceiling_usd_per_hr=1.50)
    assert selection.chosen.gpu_type_id == "NVIDIA A40"


def test_create_pod_returns_error_detail_on_non_2xx():
    transport = FakePodTransport()
    transport.post_pods = [TransportResponse(401, {"error": "unauthorized"})]
    pod, detail = create_pod(
        transport, "fake-key", name="n", image="img", gpu_type_id="NVIDIA A40",
    )
    assert pod is None
    assert "401" in detail


# ---------------------------------------------------------------------------
# 20. A non-capacity-shaped creation error (e.g. auth) is fatal and does not
#     keep guessing other GPUs
# ---------------------------------------------------------------------------
def test_non_capacity_create_error_is_fatal_not_retried_across_gpus():
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(401, {"error": "unauthorized"})]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.classification == POD_RUNPOD_API_ERROR
    attempted = [c for c in transport.calls if c[0] == "POST" and c[1].endswith("/pods")]
    assert len(attempted) == 1  # never tried A40/A6000/L4 after a non-capacity error
