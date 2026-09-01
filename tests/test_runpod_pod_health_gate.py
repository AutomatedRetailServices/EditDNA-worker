"""D-042: the Pod-RAW workflow's health-only gate script -- guaranteed
teardown-in-finally and summary-writing behavior, exercised against a
fake provider (no real RunPod/network calls)."""
from __future__ import annotations

import json

import pytest

import runpod_pod_health_gate as gate


class _FakeResult:
    def __init__(self, passed, classification="POD_HEALTH_PASSED"):
        self.passed = passed
        self.classification = classification
        self.elapsed_s = 12.3
        self.detail = {"pod_id": "pod-x"}


class _FakeProvider:
    instances = []

    def __init__(self, transport, config, *, existing_pod_id=None, log=None):
        self.teardown_called = False
        self.health_check_called = False
        self._pod_id = "pod-x"
        self._raise_in_health_check = False
        self._passed = True
        _FakeProvider.instances.append(self)

    @property
    def pod_id(self):
        return self._pod_id

    def health_check(self):
        self.health_check_called = True
        if self._raise_in_health_check:
            raise RuntimeError("simulated health_check crash")
        return _FakeResult(self._passed)

    def teardown(self):
        self.teardown_called = True


@pytest.fixture(autouse=True)
def _patch_provider(monkeypatch):
    _FakeProvider.instances.clear()
    monkeypatch.setattr(gate, "RunPodPodExecutionProvider", _FakeProvider)
    monkeypatch.setattr(gate, "UrllibTransport", lambda: object())


@pytest.fixture
def _env(monkeypatch, tmp_path, chdir=None):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("POD_IMAGE", "ghcr.io/example/img@sha256:deadbeef")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_health_pass_returns_zero_and_writes_summary(_env):
    exit_code = gate.main()
    assert exit_code == 0
    provider = _FakeProvider.instances[0]
    assert provider.teardown_called is True
    summary = json.loads((_env / "pod-health-summary.json").read_text())
    assert summary["passed"] is True
    assert summary["pod_id"] == "pod-x"


def test_health_failure_returns_nonzero_but_still_tears_down(_env):
    _FakeProvider._passed_override = False

    def _init(self, transport, config, *, existing_pod_id=None, log=None):
        self.teardown_called = False
        self._pod_id = "pod-x"
        self._raise_in_health_check = False
        self._passed = False
        _FakeProvider.instances.append(self)

    _FakeProvider.__init__ = _init
    exit_code = gate.main()
    assert exit_code == 1
    assert _FakeProvider.instances[0].teardown_called is True


def test_teardown_still_runs_when_health_check_raises(_env):
    def _init(self, transport, config, *, existing_pod_id=None, log=None):
        self.teardown_called = False
        self._pod_id = "pod-x"
        self._raise_in_health_check = True
        self._passed = True
        _FakeProvider.instances.append(self)

    _FakeProvider.__init__ = _init

    with pytest.raises(RuntimeError, match="simulated health_check crash"):
        gate.main()

    assert _FakeProvider.instances[0].teardown_called is True


def test_existing_pod_id_env_var_is_passed_through(_env, monkeypatch):
    monkeypatch.setenv("EXISTING_POD_ID", "pod-reuse-me")
    captured = {}

    def _init(self, transport, config, *, existing_pod_id=None, log=None):
        captured["existing_pod_id"] = existing_pod_id
        self.teardown_called = False
        self._pod_id = existing_pod_id
        self._raise_in_health_check = False
        self._passed = True
        _FakeProvider.instances.append(self)

    _FakeProvider.__init__ = _init
    gate.main()
    assert captured["existing_pod_id"] == "pod-reuse-me"
