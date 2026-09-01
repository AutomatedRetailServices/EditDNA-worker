"""D-041: RunPod endpoint startup/health orchestration -- infrastructure only,
no CutSell editorial logic exercised here. Covers the 9 required scenarios
via a fully scripted fake Transport + fake clock -- no network, no RunPod
credentials, no paid GPU.
"""
from __future__ import annotations

from runpod_orchestration import (
    CAPACITY_UNAVAILABLE,
    ENDPOINT_TRANSITION_RACE,
    HEALTH_APP_FAILURE,
    HEALTH_PASSED,
    RUNPOD_API_ERROR,
    WORKER_PROVISIONING_STALLED,
    AttemptResult,
    EndpointReadiness,
    HealthOutcome,
    TransportResponse,
    cancel_job_if_active,
    run_with_bounded_retry,
    submit_and_poll_health,
    wait_for_endpoint_ready,
)


def _noop_log(event):
    pass


class FakeTransport:
    """Scripted fake: each call pops the next queued response for its
    endpoint category, in order. An empty/mismatched queue raises loudly
    rather than silently returning something wrong."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self._get_endpoint: list[TransportResponse] = []
        self._post_run: list[TransportResponse] = []
        self._get_status: list[TransportResponse] = []
        self._post_cancel: list[TransportResponse] = []

    def request(self, method, url, *, headers, json_body=None):
        self.calls.append((method, url, json_body))
        if method == "GET" and "/status/" in url:
            return self._pop(self._get_status, f"GET status ({url})")
        if method == "GET" and "/endpoints/" in url:
            return self._pop(self._get_endpoint, f"GET endpoint ({url})")
        if method == "PATCH":
            return TransportResponse(200, {})
        if method == "POST" and url.endswith("/run"):
            return self._pop(self._post_run, f"POST run ({url})")
        if method == "POST" and "/cancel/" in url:
            return self._pop(self._post_cancel, f"POST cancel ({url})")
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


# ---------------------------------------------------------------------------
# 1. endpoint becomes ready normally
# ---------------------------------------------------------------------------
def test_endpoint_becomes_ready_normally():
    transport = FakeTransport()
    transport._get_endpoint = [
        TransportResponse(200, {"templateId": "tmpl-1", "workersMin": 0, "workersMax": 1, "gpuIds": "NVIDIA A100"})
    ]
    clock = FakeClock()
    readiness = wait_for_endpoint_ready(
        transport, "ep1", "key", "tmpl-1", 1, now=clock.now, sleep=clock.sleep, log=_noop_log
    )
    assert readiness.ready is True
    assert readiness.template_id == "tmpl-1"
    assert readiness.gpu_type == "NVIDIA A100"
    assert readiness.classification is None


# ---------------------------------------------------------------------------
# 2. 409 during transition (recovers), and 409 persisting past the readiness
#    timeout (does not recover -- classified ENDPOINT_TRANSITION_RACE)
# ---------------------------------------------------------------------------
def test_409_during_transition_then_ready():
    transport = FakeTransport()
    transport._get_endpoint = [
        TransportResponse(409, None),
        TransportResponse(409, None),
        TransportResponse(200, {"templateId": "tmpl-1", "workersMax": 1}),
    ]
    clock = FakeClock()
    readiness = wait_for_endpoint_ready(
        transport, "ep1", "key", "tmpl-1", 1, poll_interval_s=5, now=clock.now, sleep=clock.sleep, log=_noop_log
    )
    assert readiness.ready is True


def test_409_persisting_past_readiness_timeout_classifies_transition_race():
    transport = FakeTransport()
    transport._get_endpoint = [TransportResponse(409, None)] * 10
    clock = FakeClock()
    readiness = wait_for_endpoint_ready(
        transport,
        "ep1",
        "key",
        "tmpl-1",
        1,
        readiness_timeout_s=30,
        poll_interval_s=10,
        now=clock.now,
        sleep=clock.sleep,
        log=_noop_log,
    )
    assert readiness.ready is False
    assert readiness.classification == ENDPOINT_TRANSITION_RACE


def test_endpoint_config_never_matches_expected_roll_classifies_capacity_unavailable():
    # No 409s at all -- the endpoint answers fine, it just never reflects the
    # roll we asked for (stale templateId/workersMax) within the readiness
    # window. This is the "endpoint transition not ready" case without an
    # explicit API rejection, distinct from a 409 race.
    transport = FakeTransport()
    transport._get_endpoint = [TransportResponse(200, {"templateId": "old-tmpl", "workersMax": 0})] * 10
    clock = FakeClock()
    readiness = wait_for_endpoint_ready(
        transport,
        "ep1",
        "key",
        "tmpl-new",
        1,
        readiness_timeout_s=20,
        poll_interval_s=10,
        now=clock.now,
        sleep=clock.sleep,
        log=_noop_log,
    )
    assert readiness.ready is False
    assert readiness.classification == CAPACITY_UNAVAILABLE


# ---------------------------------------------------------------------------
# 3. health accepted then remains IN_QUEUE (this is exactly what RAW
#    33453836301 hit -- the old code waited the full 20-minute deadline
#    blind; this must fail fast and classify it, not silently keep going).
# ---------------------------------------------------------------------------
def test_health_accepted_then_remains_in_queue_fails_fast_on_first_attempt():
    transport = FakeTransport()
    transport._post_run = [TransportResponse(200, {"id": "job-1"})]
    transport._get_status = [TransportResponse(200, {"status": "IN_QUEUE"})] * 20
    clock = FakeClock()
    outcome = submit_and_poll_health(
        transport,
        "ep1",
        "key",
        queue_grace_s=30,
        queue_stall_s=60,
        is_retry_attempt=False,
        poll_interval_s=10,
        now=clock.now,
        sleep=clock.sleep,
        log=_noop_log,
    )
    assert outcome.passed is False
    assert outcome.classification == WORKER_PROVISIONING_STALLED
    assert outcome.job_id == "job-1"
    # Failed fast at the stall threshold, not at some much larger blind
    # deadline -- this is the whole point of the fix.
    assert outcome.time_in_queue_s == 60


# ---------------------------------------------------------------------------
# 4. worker eventually starts and health completes successfully
# ---------------------------------------------------------------------------
def test_worker_eventually_starts_and_health_completes():
    transport = FakeTransport()
    transport._post_run = [TransportResponse(200, {"id": "job-2"})]
    transport._get_status = [
        TransportResponse(200, {"status": "IN_QUEUE"}),
        TransportResponse(200, {"status": "IN_QUEUE"}),
        TransportResponse(200, {"status": "IN_PROGRESS"}),
        TransportResponse(200, {"status": "COMPLETED", "output": {"ok": True, "cuda_available": True}}),
    ]
    clock = FakeClock()
    outcome = submit_and_poll_health(
        transport,
        "ep1",
        "key",
        queue_grace_s=5,
        queue_stall_s=300,
        poll_interval_s=5,
        now=clock.now,
        sleep=clock.sleep,
        log=_noop_log,
    )
    assert outcome.passed is True
    assert outcome.classification == HEALTH_PASSED


def test_health_job_runs_and_reports_a_real_app_failure_not_a_queue_problem():
    # A genuine terminal failure from RunPod itself (the job actually ran)
    # must never be reclassified as a queue/capacity issue.
    transport = FakeTransport()
    transport._post_run = [TransportResponse(200, {"id": "job-3"})]
    transport._get_status = [TransportResponse(200, {"status": "FAILED", "error": "cuda oom"})]
    clock = FakeClock()
    outcome = submit_and_poll_health(transport, "ep1", "key", now=clock.now, sleep=clock.sleep, log=_noop_log)
    assert outcome.passed is False
    assert outcome.classification == HEALTH_APP_FAILURE


def test_health_completed_but_output_not_ok_is_app_failure_not_queue_problem():
    transport = FakeTransport()
    transport._post_run = [TransportResponse(200, {"id": "job-4"})]
    transport._get_status = [
        TransportResponse(200, {"status": "COMPLETED", "output": {"ok": True, "cuda_available": False}})
    ]
    clock = FakeClock()
    outcome = submit_and_poll_health(transport, "ep1", "key", now=clock.now, sleep=clock.sleep, log=_noop_log)
    assert outcome.passed is False
    assert outcome.classification == HEALTH_APP_FAILURE


def test_health_submit_http_error_classifies_runpod_api_error():
    transport = FakeTransport()
    transport._post_run = [TransportResponse(500, None)]
    clock = FakeClock()
    outcome = submit_and_poll_health(transport, "ep1", "key", now=clock.now, sleep=clock.sleep, log=_noop_log)
    assert outcome.classification == RUNPOD_API_ERROR
    assert outcome.job_id is None


# ---------------------------------------------------------------------------
# 5. capacity stall exits early (second/retry attempt -> persistent,
#    CAPACITY_UNAVAILABLE rather than the first-attempt "maybe transient"
#    WORKER_PROVISIONING_STALLED label)
# ---------------------------------------------------------------------------
def test_capacity_stall_on_retry_attempt_classifies_capacity_unavailable():
    transport = FakeTransport()
    transport._post_run = [TransportResponse(200, {"id": "job-5"})]
    transport._get_status = [TransportResponse(200, {"status": "IN_QUEUE"})] * 20
    clock = FakeClock()
    outcome = submit_and_poll_health(
        transport,
        "ep1",
        "key",
        queue_stall_s=60,
        is_retry_attempt=True,
        poll_interval_s=10,
        now=clock.now,
        sleep=clock.sleep,
        log=_noop_log,
    )
    assert outcome.passed is False
    assert outcome.classification == CAPACITY_UNAVAILABLE


# ---------------------------------------------------------------------------
# 6 / 7. bounded infra retry succeeds / bounded infra retry exhausted
# ---------------------------------------------------------------------------
def test_bounded_infra_retry_succeeds_on_second_attempt():
    calls = {"n": 0}
    roll_calls: list[bool] = []

    def roll():
        roll_calls.append(True)

    def attempt(is_retry: bool) -> AttemptResult:
        calls["n"] += 1
        readiness = EndpointReadiness(True, "tmpl", 0, 1, "A100", 5.0)
        if calls["n"] == 1:
            health = HealthOutcome(WORKER_PROVISIONING_STALLED, False, "job-1", 300.0, "IN_QUEUE")
        else:
            health = HealthOutcome(HEALTH_PASSED, True, "job-2", 5.0, "COMPLETED")
        return AttemptResult(readiness=readiness, health=health)

    clock = FakeClock()
    result = run_with_bounded_retry(
        roll, attempt, max_infra_retries=1, backoff_s=15, sleep=clock.sleep, now=clock.now, log=_noop_log
    )
    assert result.passed is True
    assert result.classification == HEALTH_PASSED
    assert len(result.attempts) == 2
    # Rolled again exactly once, before the retry -- never before the first
    # attempt (that roll already happened by the time this runs).
    assert len(roll_calls) == 1


def test_bounded_infra_retry_exhausted_stops_after_the_ceiling():
    calls = {"n": 0}

    def roll():
        pass

    def attempt(is_retry: bool) -> AttemptResult:
        calls["n"] += 1
        readiness = EndpointReadiness(True, "tmpl", 0, 1, "A100", 5.0)
        classification = CAPACITY_UNAVAILABLE if is_retry else WORKER_PROVISIONING_STALLED
        health = HealthOutcome(classification, False, "job", 300.0, "IN_QUEUE")
        return AttemptResult(readiness=readiness, health=health)

    clock = FakeClock()
    result = run_with_bounded_retry(
        roll, attempt, max_infra_retries=1, backoff_s=15, sleep=clock.sleep, now=clock.now, log=_noop_log
    )
    assert result.passed is False
    assert result.classification == CAPACITY_UNAVAILABLE
    assert len(result.attempts) == 2
    # No infinite loop: the ceiling is 1 + max_infra_retries, never a third.
    assert calls["n"] == 2


def test_health_app_failure_is_never_retried():
    calls = {"n": 0}

    def roll():
        pass

    def attempt(is_retry: bool) -> AttemptResult:
        calls["n"] += 1
        readiness = EndpointReadiness(True, "tmpl", 0, 1, "A100", 5.0)
        health = HealthOutcome(HEALTH_APP_FAILURE, False, "job", 1.0, "FAILED")
        return AttemptResult(readiness=readiness, health=health)

    clock = FakeClock()
    result = run_with_bounded_retry(
        roll, attempt, max_infra_retries=1, backoff_s=15, sleep=clock.sleep, now=clock.now, log=_noop_log
    )
    assert result.passed is False
    assert result.classification == HEALTH_APP_FAILURE
    # A real application/CUDA failure is not a flake -- never retried.
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 8. teardown always runs -- a stalled job is cancelled before the retry,
#    and the full attempt flow (readiness + health + teardown between
#    attempts) is exercised end to end.
# ---------------------------------------------------------------------------
def test_teardown_cancels_a_job_still_in_queue():
    transport = FakeTransport()
    transport._get_status = [TransportResponse(200, {"status": "IN_QUEUE"})]
    transport._post_cancel = [TransportResponse(200, {})]
    cancel_job_if_active(transport, "ep1", "key", "job-1", log=_noop_log)
    assert any(c[0] == "POST" and "/cancel/" in c[1] for c in transport.calls)


def test_teardown_skips_cancel_when_job_already_terminal():
    transport = FakeTransport()
    transport._get_status = [TransportResponse(200, {"status": "COMPLETED"})]
    cancel_job_if_active(transport, "ep1", "key", "job-1", log=_noop_log)
    assert not any(c[0] == "POST" and "/cancel/" in c[1] for c in transport.calls)


def test_teardown_is_a_no_op_when_no_job_was_ever_submitted():
    transport = FakeTransport()
    cancel_job_if_active(transport, "ep1", "key", None, log=_noop_log)
    assert transport.calls == []


def test_full_attempt_flow_tears_down_stalled_job_before_a_successful_retry():
    transport = FakeTransport()
    transport._get_endpoint = [
        TransportResponse(200, {"templateId": "tmpl", "workersMax": 1}),
        TransportResponse(200, {"templateId": "tmpl", "workersMax": 1}),
    ]
    transport._post_run = [TransportResponse(200, {"id": "job-stalled"}), TransportResponse(200, {"id": "job-ok"})]
    transport._get_status = [
        TransportResponse(200, {"status": "IN_QUEUE"}),  # t=0
        TransportResponse(200, {"status": "IN_QUEUE"}),  # t=10
        TransportResponse(200, {"status": "IN_QUEUE"}),  # t=20 -> stalls (queue_stall_s=20)
        TransportResponse(200, {"status": "IN_QUEUE"}),  # cancel_job_if_active's own status check
        TransportResponse(200, {"status": "COMPLETED", "output": {"ok": True, "cuda_available": True}}),
    ]
    transport._post_cancel = [TransportResponse(200, {})]
    clock = FakeClock()

    def roll():
        pass

    def attempt(is_retry: bool) -> AttemptResult:
        readiness = wait_for_endpoint_ready(
            transport, "ep1", "key", "tmpl", 1, now=clock.now, sleep=clock.sleep, log=_noop_log
        )
        health = submit_and_poll_health(
            transport,
            "ep1",
            "key",
            queue_stall_s=20,
            poll_interval_s=10,
            is_retry_attempt=is_retry,
            now=clock.now,
            sleep=clock.sleep,
            log=_noop_log,
        )
        if not health.passed:
            cancel_job_if_active(transport, "ep1", "key", health.job_id, log=_noop_log)
        return AttemptResult(readiness=readiness, health=health)

    result = run_with_bounded_retry(
        roll, attempt, max_infra_retries=1, backoff_s=1, sleep=clock.sleep, now=clock.now, log=_noop_log
    )
    assert result.passed is True
    assert len(result.attempts) == 2
    assert result.attempts[0].health.classification == WORKER_PROVISIONING_STALLED
    # The stalled first job was torn down before the retry ran.
    assert any(c[0] == "POST" and "/cancel/" in c[1] for c in transport.calls)


# ---------------------------------------------------------------------------
# 9. Video00 never submits before health PASS -- pins the exit-code/`passed`
#    contract the workflow step's skip-on-failure behavior depends on.
# ---------------------------------------------------------------------------
def test_orchestration_result_passed_is_false_whenever_health_did_not_actually_pass():
    def roll():
        pass

    def attempt(is_retry: bool) -> AttemptResult:
        readiness = EndpointReadiness(True, "tmpl", 0, 1, "A100", 1.0)
        health = HealthOutcome(HEALTH_APP_FAILURE, False, "job", 1.0, "FAILED")
        return AttemptResult(readiness=readiness, health=health)

    result = run_with_bounded_retry(
        roll, attempt, max_infra_retries=0, sleep=lambda s: None, now=lambda: 0.0, log=_noop_log
    )
    assert result.passed is False
    # This is exactly what main() turns into the step's exit code: 0 only if
    # passed, 1 otherwise -- and GitHub Actions skips the next (non-`if:
    # always()`) step, "Submit original six-minute Video00", on a nonzero exit.
    assert (0 if result.passed else 1) == 1
