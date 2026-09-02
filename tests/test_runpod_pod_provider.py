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
    fetch_pod_logs,
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
        self.get_logs: list[TransportResponse] = []

    def request(self, method, url, *, headers, json_body=None):
        self.calls.append((method, url, json_body))
        if method == "GET" and ("/gpuTypes" in url or "/gpu-types" in url):
            return self._pop(self.get_gpu_types, f"GET gpuTypes ({url})")
        if method == "GET" and url.endswith("/logs"):
            return self._pop(self.get_logs, f"GET logs ({url})")
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
        TransportResponse(200, {"id": "pod-2", "status": "RUNNING", "imageName": "img"}),  # fresh pod-2 becomes ready
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
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "old-image"}),
        TransportResponse(200, {"id": "pod-2", "status": "RUNNING", "imageName": "new-image"}),
    ]
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
    transport.get_pod = [TransportResponse(200, {"id": "pod-new", "status": "RUNNING"})]
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
    transport.get_pod = [TransportResponse(200, {"id": "pod-a6000", "status": "RUNNING"})]
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
    transport.get_pod = [TransportResponse(200, {"id": "pod-l4", "status": "RUNNING"})]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.pod_id == "pod-l4"
    assert result.gpu_selection.chosen.gpu_type_id == "NVIDIA L4"


def test_community_exhausted_falls_back_to_secure_cloud_sweep():
    # D-042's own first live test shape: every approved GPU rejected under
    # COMMUNITY, but one succeeds once the SECURE sweep starts.
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [
        TransportResponse(400, {"error": "no instances currently available"}),  # COMMUNITY 4090
        TransportResponse(400, {"error": "no instances currently available"}),  # COMMUNITY A40
        TransportResponse(400, {"error": "no instances currently available"}),  # COMMUNITY A6000
        TransportResponse(400, {"error": "no instances currently available"}),  # COMMUNITY L4
        TransportResponse(201, {"id": "pod-secure-4090"}),  # SECURE 4090 succeeds
    ]
    transport.get_pod = [TransportResponse(200, {"id": "pod-secure-4090", "status": "RUNNING"})]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.classification == POD_CREATED_FRESH
    assert result.pod_id == "pod-secure-4090"
    assert result.gpu_selection.chosen.gpu_type_id == "NVIDIA GeForce RTX 4090"
    assert result.detail["cloud_type"] == "SECURE"
    cloud_sequence = [c["cloudType"] for _m, _u, c in transport.calls if _m == "POST" and _u.endswith("/pods")]
    assert cloud_sequence == ["COMMUNITY", "COMMUNITY", "COMMUNITY", "COMMUNITY", "SECURE"]


# ---------------------------------------------------------------------------
# 9. No compatible GPU available -> CAPACITY_UNAVAILABLE, nothing provisioned
# ---------------------------------------------------------------------------
def test_no_compatible_gpu_available_reports_capacity_unavailable():
    # Must exhaust both the COMMUNITY sweep AND the SECURE sweep (D-042's
    # first live test hit exactly this shape -- "no instances currently
    # available" on all 4 approved GPUs under COMMUNITY alone -- before
    # concluding capacity is genuinely unavailable everywhere.
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [
        TransportResponse(400, {"error": "no instances available"}),
    ] * (2 * len(APPROVED_POD_GPU_TYPE_IDS))
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.classification == POD_CAPACITY_UNAVAILABLE
    assert result.pod_id is None
    attempted_clouds = {c["cloudType"] for _m, _u, c in transport.calls if _m == "POST" and _u.endswith("/pods")}
    assert attempted_clouds == {"COMMUNITY", "SECURE"}


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
    transport.get_pod = [TransportResponse(200, {"id": "pod-a40", "status": "RUNNING"})]
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
    transport.get_pod = [TransportResponse(200, {"id": "pod-4090", "status": "RUNNING"})]
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


def test_fresh_create_waits_for_running_before_returning():
    # D-042's own first successful live Pod creation: the API accepting a
    # create call is not the same as the Pod being RUNNING. A fresh create
    # must poll until RUNNING before handing back POD_CREATED_FRESH.
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-new"})]
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-new", "status": "PENDING"}),
        TransportResponse(200, {"id": "pod-new", "status": "RUNNING"}),
    ]
    provider, _clock = _provider(transport, existing_pod_id=None)

    result = provider.ensure_ready()

    assert result.classification == POD_CREATED_FRESH
    assert result.pod_id == "pod-new"
    get_pod_calls = [c for c in transport.calls if c[0] == "GET" and "/pods/" in c[1]]
    assert len(get_pod_calls) == 2


def test_fresh_create_that_never_becomes_running_is_reported_not_silently_passed():
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-new"})]
    still_pending = TransportResponse(200, {"id": "pod-new", "status": "PENDING"})
    transport.get_pod = [still_pending, still_pending, still_pending]
    provider, _clock = _provider(transport, existing_pod_id=None, restart_wait_timeout_s=10.0, poll_interval_s=5.0)
    provider._cfg.create_wait_timeout_s = 10.0

    result = provider.ensure_ready()

    assert result.classification == POD_RESTART_UNAVAILABLE
    assert result.pod_id == "pod-new"  # the Pod exists -- caller must still be able to stop it


def test_health_poll_retries_through_proxy_403_until_app_answers():
    # RunPod's proxy 403s a Pod whose container hasn't started listening
    # yet -- confirmed live. The health poll must retry through this, not
    # report a false failure on the first attempt.
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"})]
    calls = []

    def _http_get(url):
        calls.append(url)
        if len(calls) < 3:
            return 403, None
        return 200, {"ok": True, "cuda_available": True}

    provider, clock = _provider(
        transport, existing_pod_id="pod-1", image="img", http_get=_http_get,
    )
    provider._cfg.health_poll_interval_s = 1.0
    provider._cfg.health_poll_timeout_s = 60.0

    result = provider.health_check()

    assert result.passed is True
    assert len(calls) == 3


def test_health_poll_gives_up_after_bounded_timeout_still_403():
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"})]
    provider, clock = _provider(
        transport, existing_pod_id="pod-1", image="img", http_get=lambda url: (403, None),
    )
    provider._cfg.health_poll_interval_s = 5.0
    provider._cfg.health_poll_timeout_s = 12.0

    result = provider.health_check()

    assert result.passed is False
    assert result.classification == POD_HEALTH_APP_FAILURE
    assert result.detail["health_status_code"] == 403


def test_health_poll_does_not_retry_a_real_app_failure_response():
    # A genuine, well-formed app answer (even a failing one) must return
    # immediately -- never mistaken for "still booting".
    transport = FakePodTransport()
    transport.get_pod = [TransportResponse(200, {"id": "pod-1", "status": "RUNNING", "imageName": "img"})]
    calls = []
    provider, clock = _provider(
        transport,
        existing_pod_id="pod-1",
        image="img",
        http_get=lambda url: calls.append(url) or (200, {"ok": True, "cuda_available": False}),
    )

    result = provider.health_check()

    assert result.passed is False
    assert result.classification == POD_HEALTH_APP_FAILURE
    assert len(calls) == 1  # no retry -- this was a real answer


def test_health_check_never_reaches_video00_stage_when_lifecycle_fails():
    # ensure_ready() itself failing (no pod_id) must short-circuit
    # health_check before any HTTP GET to the pod is attempted.
    calls = []
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(400, {"error": "no instances available"})] * (2 * len(APPROVED_POD_GPU_TYPE_IDS))
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
    transport.get_pod = [
        TransportResponse(200, {"id": "pod-1", "status": "ERROR", "imageName": "img"}),
        TransportResponse(200, {"id": "pod-2", "status": "RUNNING", "imageName": "img"}),
    ]
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
    # Even if every single GPU creation attempt fails (across both cloud
    # sweeps), ensure_ready must return (CAPACITY_UNAVAILABLE) rather than
    # looping back to re-inspect pod-1 again.
    transport.post_pods = [TransportResponse(400, {"error": "no instances available"})] * (2 * len(APPROVED_POD_GPU_TYPE_IDS))
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


def test_fetch_pod_logs_returns_first_non_404_response():
    transport = FakePodTransport()
    transport.get_logs = [TransportResponse(404, None), TransportResponse(200, {"lines": ["hello"]})]
    url, status, body = fetch_pod_logs(transport, "fake-key", "pod-1", log=_noop_log)
    assert status == 200
    assert body == {"lines": ["hello"]}
    assert url.endswith("/logs")


def test_fetch_pod_logs_returns_last_result_when_all_candidates_404():
    transport = FakePodTransport()
    transport.get_logs = [TransportResponse(404, None), TransportResponse(404, None)]
    _url, status, body = fetch_pod_logs(transport, "fake-key", "pod-1", log=_noop_log)
    assert status == 404
    assert body is None


def test_fetch_pod_logs_treats_route_not_found_400_like_404():
    # Confirmed live: RunPod REST v1 answers an unknown route with HTTP 400
    # (not 404), body naming the path and "does not exist in the
    # specification" -- functionally identical to 404 for this purpose.
    transport = FakePodTransport()
    route_not_found = [{"error": "At #/paths/get for GET .../logs, ... does not exist in the specification"}]
    transport.get_logs = [TransportResponse(400, route_not_found), TransportResponse(200, {"lines": ["hi"]})]
    url, status, body = fetch_pod_logs(transport, "fake-key", "pod-1", log=_noop_log)
    assert status == 200
    assert body == {"lines": ["hi"]}


def test_fetch_pod_logs_does_not_swallow_a_real_400():
    # A genuine 400 (bad request, auth, validation) must be returned as-is,
    # never silently treated as "try the next candidate".
    transport = FakePodTransport()
    transport.get_logs = [TransportResponse(400, {"error": "unauthorized"})]
    url, status, body = fetch_pod_logs(transport, "fake-key", "pod-1", log=_noop_log)
    assert status == 400
    assert body == {"error": "unauthorized"}


def test_create_pod_returns_error_detail_on_non_2xx():
    transport = FakePodTransport()
    transport.post_pods = [TransportResponse(401, {"error": "unauthorized"})]
    pod, detail = create_pod(
        transport, "fake-key", name="n", image="img", gpu_type_id="NVIDIA A40",
    )
    assert pod is None
    assert "401" in detail


def test_create_pod_sends_ports_and_docker_start_cmd_as_arrays():
    # Regression: D-042's first live health-only test hit a real RunPod
    # 400 -- "got string, want array" for both `ports` and
    # `dockerStartCmd`. RunPod's REST v1 schema requires JSON arrays for
    # both fields, never single strings.
    transport = FakePodTransport()
    transport.post_pods = [TransportResponse(201, {"id": "pod-1"})]
    create_pod(
        transport,
        "fake-key",
        name="n",
        image="img",
        gpu_type_id="NVIDIA A40",
        start_command="python3 -m cutsell_worker.pod_job_server",
    )
    _method, _url, body = transport.calls[0]
    assert isinstance(body["ports"], list) and body["ports"] == ["8080/http"]
    assert isinstance(body["dockerStartCmd"], list)
    assert body["dockerStartCmd"] == ["python3", "-m", "cutsell_worker.pod_job_server"]


def test_create_pod_ports_list_passthrough_and_no_start_command_omits_field():
    transport = FakePodTransport()
    transport.post_pods = [TransportResponse(201, {"id": "pod-1"})]
    create_pod(
        transport, "fake-key", name="n", image="img", gpu_type_id="NVIDIA A40",
        ports=["8080/http", "22/tcp"],
    )
    _method, _url, body = transport.calls[0]
    assert body["ports"] == ["8080/http", "22/tcp"]
    assert "dockerStartCmd" not in body


def test_create_pod_with_template_id_sends_minimal_payload_only():
    # D-042 follow-up: creating a Pod FROM the CutSell-Pod-QA template
    # (rather than an inline ad-hoc config) must send only the
    # Pod-instance-specific fields -- image/ports/env/dockerStartCmd/disk
    # all come from the template itself and must never be duplicated or
    # guessed here.
    transport = FakePodTransport()
    transport.post_pods = [TransportResponse(201, {"id": "pod-1"})]
    create_pod(
        transport,
        "fake-key",
        name="n",
        image="img-ignored-in-template-mode",
        gpu_type_id="NVIDIA A40",
        start_command="python3 -m cutsell_worker.pod_job_server",
        env={"IGNORED": "1"},
        ports=["22/tcp"],
        template_id="5moabglc4m",
    )
    _method, _url, body = transport.calls[0]
    assert body == {
        "name": "n",
        "templateId": "5moabglc4m",
        "gpuTypeIds": ["NVIDIA A40"],
        "gpuCount": 1,
        "cloudType": "COMMUNITY",
    }
    for field in ("imageName", "ports", "env", "dockerStartCmd", "containerDiskInGb"):
        assert field not in body


def test_create_pod_without_template_id_is_unaffected_inline_mode():
    # Regression lock: passing template_id=None (the default) must produce
    # exactly the same inline payload shape as before this parameter
    # existed -- no behavior change for the existing inline-config path.
    transport = FakePodTransport()
    transport.post_pods = [TransportResponse(201, {"id": "pod-1"})]
    create_pod(
        transport,
        "fake-key",
        name="n",
        image="img",
        gpu_type_id="NVIDIA A40",
        start_command="python3 -m cutsell_worker.pod_job_server",
    )
    _method, _url, body = transport.calls[0]
    assert body["imageName"] == "img"
    assert "templateId" not in body


def test_pod_execution_config_env_is_forwarded_to_create_pod():
    # D-042 follow-up ("restore the known-working execution model"): a real
    # gap -- PodExecutionConfig.env did not exist, so create_pod()'s own
    # `env` parameter was never reached through the provider's fresh-create
    # path; every earlier inline-mode live Pod test this session ran with
    # an empty env. The direct-execution benchmark path needs a populated
    # one (CUTSELL_BENCHMARK_PAYLOAD_JSON at minimum).
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-new"})]
    transport.get_pod = [TransportResponse(200, {"id": "pod-new", "status": "RUNNING"})]
    provider, _clock = _provider(transport, existing_pod_id=None, env={"CUTSELL_BENCHMARK_PAYLOAD_JSON": "{}"})

    result = provider.ensure_ready()

    assert result.classification == POD_CREATED_FRESH
    post_call = next(c for c in transport.calls if c[0] == "POST" and c[1].endswith("/pods"))
    assert post_call[2]["env"] == {"CUTSELL_BENCHMARK_PAYLOAD_JSON": "{}"}


def test_pod_execution_config_env_defaults_to_empty_when_unset():
    transport = FakePodTransport()
    transport.get_gpu_types = [TransportResponse(200, [])]
    transport.post_pods = [TransportResponse(201, {"id": "pod-new"})]
    transport.get_pod = [TransportResponse(200, {"id": "pod-new", "status": "RUNNING"})]
    provider, _clock = _provider(transport, existing_pod_id=None)

    provider.ensure_ready()

    post_call = next(c for c in transport.calls if c[0] == "POST" and c[1].endswith("/pods"))
    assert post_call[2]["env"] == {}


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
