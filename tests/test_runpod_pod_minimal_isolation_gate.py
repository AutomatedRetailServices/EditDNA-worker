"""D-042 follow-up ("FINAL POD EXECUTION ISOLATION -- MINIMAL KNOWN-GOOD
IMAGE"): the minimal public-image Pod execution isolation gate. Fully
mocked: no network, no GPU, no real RunPod API calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import runpod_pod_minimal_isolation_gate as gate


# ---------------------------------------------------------------------------
# build_minimal_isolation_config -- pure function
# ---------------------------------------------------------------------------
def test_build_minimal_isolation_config_forwards_everything():
    config = gate.build_minimal_isolation_config(
        api_key="k",
        pod_name="n",
        image="nvidia/cuda:12.4.1-base-ubuntu22.04",
        start_command="sh -c 'echo hi'",
        container_disk_gb=20,
        cost_ceiling_usd_per_hr=1.5,
        cloud_types=("SECURE",),
    )
    assert config.image == "nvidia/cuda:12.4.1-base-ubuntu22.04"
    assert config.start_command == "sh -c 'echo hi'"
    assert config.container_disk_gb == 20
    assert config.cloud_types == ("SECURE",)
    assert config.env is None  # no CutSell dependency of any kind
    assert config.template_id is None  # never template mode -- a public image, not CutSell-Pod-QA


# ---------------------------------------------------------------------------
# collect_pod_snapshots
# ---------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_collect_pod_snapshots_returns_exactly_one_when_window_is_zero():
    clock = FakeClock()
    calls = []

    def get_pod_fn(transport, api_key, pod_id):
        calls.append(pod_id)
        return {"id": pod_id, "machine": {}}

    snapshots = gate.collect_pod_snapshots(
        object(), "key", "pod-1", window_s=0, interval_s=60, now=clock.now, sleep=clock.sleep, get_pod_fn=get_pod_fn,
    )
    assert len(snapshots) == 1
    assert len(calls) == 1


def test_collect_pod_snapshots_takes_multiple_snapshots_over_the_window():
    clock = FakeClock()

    def get_pod_fn(transport, api_key, pod_id):
        return {"id": pod_id, "machine": {}}

    snapshots = gate.collect_pod_snapshots(
        object(), "key", "pod-1", window_s=30, interval_s=10, now=clock.now, sleep=clock.sleep, get_pod_fn=get_pod_fn,
    )
    # 0s, 10s, 20s, 30s -- four checks before window_s is reached/exceeded
    assert len(snapshots) == 4
    assert snapshots[-1]["elapsed_s"] >= 30


def test_collect_pod_snapshots_records_error_and_keeps_going():
    clock = FakeClock()
    call_count = {"n": 0}

    def get_pod_fn(transport, api_key, pod_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient GET failure")
        return {"id": pod_id, "machine": {"gpuId": "x"}}

    snapshots = gate.collect_pod_snapshots(
        object(), "key", "pod-1", window_s=15, interval_s=10, now=clock.now, sleep=clock.sleep, get_pod_fn=get_pod_fn,
    )
    assert "error" in snapshots[0]
    assert snapshots[0]["error"] == "transient GET failure"
    assert "pod" in snapshots[1]
    assert snapshots[1]["pod"]["machine"] == {"gpuId": "x"}


def test_collect_pod_snapshots_redacts_env_values():
    clock = FakeClock()

    def get_pod_fn(transport, api_key, pod_id):
        return {"id": pod_id, "env": {"SOME_KEY": "real-value"}}

    snapshots = gate.collect_pod_snapshots(
        object(), "key", "pod-1", window_s=1, interval_s=10, now=clock.now, sleep=clock.sleep, get_pod_fn=get_pod_fn,
    )
    assert snapshots[0]["pod"]["env"]["SOME_KEY"] == "<redacted>"


# ---------------------------------------------------------------------------
# machine_ever_populated
# ---------------------------------------------------------------------------
def test_machine_ever_populated_true_when_any_snapshot_has_content():
    snapshots = [
        {"pod": {"machine": {}}},
        {"pod": {"machine": {"gpuDisplayName": "RTX 4090"}}},
    ]
    assert gate.machine_ever_populated(snapshots) is True


def test_machine_ever_populated_false_when_always_empty():
    snapshots = [{"pod": {"machine": {}}}, {"pod": {"machine": {}}}]
    assert gate.machine_ever_populated(snapshots) is False


def test_machine_ever_populated_false_on_error_only_snapshots():
    snapshots = [{"error": "boom"}, {"error": "boom again"}]
    assert gate.machine_ever_populated(snapshots) is False


def test_machine_ever_populated_false_when_pod_is_none():
    snapshots = [{"pod": None}]
    assert gate.machine_ever_populated(snapshots) is False


# ---------------------------------------------------------------------------
# log_confirms_execution
# ---------------------------------------------------------------------------
def test_log_confirms_execution_true_for_matching_string():
    assert gate.log_confirms_execution("stdout: POD_EXECUTION_OK\n") is True


def test_log_confirms_execution_true_for_matching_dict():
    assert gate.log_confirms_execution({"lines": ["POD_EXECUTION_OK"]}) is True


def test_log_confirms_execution_false_for_unrelated_content():
    assert gate.log_confirms_execution("some other log line") is False
    assert gate.log_confirms_execution({"error": "not found"}) is False
    assert gate.log_confirms_execution(None) is False


# ---------------------------------------------------------------------------
# main() -- full orchestration, fully mocked
# ---------------------------------------------------------------------------
class _FakeLifecycleResult:
    def __init__(self, pod_id, classification="POD_CREATED_FRESH"):
        self.pod_id = pod_id
        self.classification = classification


class _FakeProvider:
    instances = []

    def __init__(self, transport, config, *, existing_pod_id=None, log=None):
        self.config = config
        self.existing_pod_id = existing_pod_id
        self.teardown_called = False
        self._pod_id = "pod-iso-1"
        self._lifecycle_result = _FakeLifecycleResult("pod-iso-1")
        _FakeProvider.instances.append(self)

    @property
    def pod_id(self):
        return self._pod_id

    def ensure_ready(self):
        return self._lifecycle_result

    def teardown(self):
        self.teardown_called = True


@pytest.fixture(autouse=True)
def _patch_provider(monkeypatch):
    _FakeProvider.instances.clear()
    monkeypatch.setattr(gate, "RunPodPodExecutionProvider", _FakeProvider)
    monkeypatch.setattr(gate, "UrllibTransport", lambda: object())


@pytest.fixture
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("OBSERVE_WINDOW_S", "0.02")
    monkeypatch.setenv("OBSERVE_POLL_INTERVAL_S", "0.01")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _patch_pod_apis(monkeypatch, *, machine=None, logs_body=None, logs_status=200):
    monkeypatch.setattr(gate, "get_pod", lambda transport, api_key, pod_id: {"id": pod_id, "machine": machine or {}})
    monkeypatch.setattr(
        gate, "fetch_pod_logs", lambda transport, api_key, pod_id, log=None: ("https://x/logs", logs_status, logs_body)
    )
    monkeypatch.setattr(gate, "delete_pod", lambda transport, api_key, pod_id: True)


def test_invalid_cloud_type_aborts_before_any_pod_action(monkeypatch, _env):
    monkeypatch.setenv("POD_ISOLATION_CLOUD_TYPE", "BOGUS")
    exit_code = gate.main()
    assert exit_code == 1
    assert _FakeProvider.instances == []


def test_lifecycle_failure_still_tears_down_and_reports(monkeypatch, _env):
    def _init(self, transport, config, *, existing_pod_id=None, log=None):
        self.config = config
        self.teardown_called = False
        self._pod_id = None
        self._lifecycle_result = _FakeLifecycleResult(None, classification="POD_CAPACITY_UNAVAILABLE")
        _FakeProvider.instances.append(self)

    monkeypatch.setattr(_FakeProvider, "__init__", _init)
    deleted_calls = []
    monkeypatch.setattr(gate, "delete_pod", lambda transport, api_key, pod_id: deleted_calls.append(pod_id) or True)

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances[0].teardown_called is True
    assert deleted_calls == []  # never deletes a Pod that was never created
    summary = json.loads((_env / "pod-minimal-isolation-summary.json").read_text())
    assert summary["classification"] == "POD_LIFECYCLE_FAILED"


def test_machine_populated_classifies_execution_confirmed(monkeypatch, _env):
    _patch_pod_apis(monkeypatch, machine={"gpuDisplayName": "RTX 4090"}, logs_body=None, logs_status=403)

    exit_code = gate.main()

    assert exit_code == 0
    assert _FakeProvider.instances[0].teardown_called is True
    summary = json.loads((_env / "pod-minimal-isolation-summary.json").read_text())
    assert summary["classification"] == "CONTAINER_EXECUTION_CONFIRMED"
    assert summary["machine_ever_populated"] is True
    assert summary["pod_deleted"] is True


def test_logs_confirm_execution_even_when_machine_stays_empty(monkeypatch, _env):
    _patch_pod_apis(monkeypatch, machine={}, logs_body="POD_EXECUTION_OK\n", logs_status=200)

    exit_code = gate.main()

    assert exit_code == 0
    summary = json.loads((_env / "pod-minimal-isolation-summary.json").read_text())
    assert summary["classification"] == "CONTAINER_EXECUTION_CONFIRMED"
    assert summary["machine_ever_populated"] is False
    assert summary["log_confirms_execution"] is True


def test_no_evidence_classifies_execution_not_confirmed(monkeypatch, _env):
    _patch_pod_apis(monkeypatch, machine={}, logs_body=None, logs_status=403)

    exit_code = gate.main()

    assert exit_code == 0  # the diagnostic itself still completed successfully
    summary = json.loads((_env / "pod-minimal-isolation-summary.json").read_text())
    assert summary["classification"] == "CONTAINER_EXECUTION_NOT_CONFIRMED"
    assert summary["machine_ever_populated"] is False
    assert summary["log_confirms_execution"] is False


def test_pod_is_deleted_after_teardown_on_success_path(monkeypatch, _env):
    delete_calls = []
    monkeypatch.setattr(gate, "get_pod", lambda transport, api_key, pod_id: {"id": pod_id, "machine": {}})
    monkeypatch.setattr(gate, "fetch_pod_logs", lambda transport, api_key, pod_id, log=None: ("url", 403, None))
    monkeypatch.setattr(gate, "delete_pod", lambda transport, api_key, pod_id: delete_calls.append(pod_id) or True)

    gate.main()

    assert delete_calls == ["pod-iso-1"]
    assert _FakeProvider.instances[0].teardown_called is True


def test_existing_pod_id_env_var_threaded_through(monkeypatch, _env):
    monkeypatch.setenv("EXISTING_POD_ID", "pod-reuse-me")
    captured = {}

    def _init(self, transport, config, *, existing_pod_id=None, log=None):
        captured["existing_pod_id"] = existing_pod_id
        self.teardown_called = False
        self._pod_id = "pod-reuse-me"
        self._lifecycle_result = _FakeLifecycleResult("pod-reuse-me")
        _FakeProvider.instances.append(self)

    monkeypatch.setattr(_FakeProvider, "__init__", _init)
    _patch_pod_apis(monkeypatch, machine={}, logs_body=None, logs_status=403)

    gate.main()

    assert captured["existing_pod_id"] == "pod-reuse-me"


def test_default_image_and_start_command_used_when_unset(monkeypatch, _env):
    _patch_pod_apis(monkeypatch, machine={}, logs_body=None, logs_status=403)

    gate.main()

    config = _FakeProvider.instances[0].config
    assert config.image == gate.DEFAULT_ISOLATION_IMAGE
    assert config.start_command == gate.DEFAULT_ISOLATION_START_COMMAND


def test_default_cloud_type_is_secure(monkeypatch, _env):
    _patch_pod_apis(monkeypatch, machine={}, logs_body=None, logs_status=403)

    gate.main()

    assert _FakeProvider.instances[0].config.cloud_types == ("SECURE",)


def test_summary_never_leaks_env_values(monkeypatch, _env):
    def get_pod_fn(transport, api_key, pod_id):
        return {"id": pod_id, "machine": {}, "env": {"AWS_SECRET_ACCESS_KEY": "not-a-real-secret-fixture-value"}}

    monkeypatch.setattr(gate, "get_pod", get_pod_fn)
    monkeypatch.setattr(
        gate,
        "fetch_pod_logs",
        lambda transport, api_key, pod_id, log=None: ("url", 200, {"env": {"GEMINI_API_KEY": "not-a-real-secret-fixture-value-2"}}),
    )
    monkeypatch.setattr(gate, "delete_pod", lambda transport, api_key, pod_id: True)

    gate.main()

    raw = (_env / "pod-minimal-isolation-summary.json").read_text()
    assert "not-a-real-secret-fixture-value" not in raw
    assert "not-a-real-secret-fixture-value-2" not in raw
