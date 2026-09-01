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


def test_diagnose_pod_id_env_var_skips_lifecycle_entirely(monkeypatch, capsys):
    # Zero-cost diagnostic mode must never construct a
    # RunPodPodExecutionProvider (no create/start) and must never require
    # POD_IMAGE to be set -- it only reads an existing Pod's state + logs.
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("DIAGNOSE_POD_LOGS_ID", "pod-already-stopped")
    monkeypatch.setattr(gate, "get_pod", lambda transport, api_key, pod_id: {"id": pod_id, "status": "EXITED"})
    monkeypatch.setattr(
        gate, "fetch_pod_logs", lambda transport, api_key, pod_id, log=None: ("https://x/logs", 200, {"lines": ["boom"]})
    )

    exit_code = gate.main()

    assert exit_code == 0
    assert _FakeProvider.instances == []  # lifecycle never touched
    out = capsys.readouterr().out
    assert "pod-already-stopped" in out
    assert "boom" in out


def test_template_action_fetch_base_never_touches_pod_lifecycle(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("TEMPLATE_ACTION", "fetch_base")
    monkeypatch.setattr(
        gate,
        "find_template_by_name",
        lambda transport, api_key, name: {"id": "07g9dovc17", "name": name, "env": {"SECRET": "real-value"}},
    )

    exit_code = gate.main()

    assert exit_code == 0
    assert _FakeProvider.instances == []
    out = capsys.readouterr().out
    assert "07g9dovc17" in out
    assert "real-value" not in out
    assert "<redacted>" in out


def test_template_action_fetch_base_reports_missing_template(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("TEMPLATE_ACTION", "fetch_base")
    monkeypatch.setattr(gate, "find_template_by_name", lambda transport, api_key, name: None)

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances == []


def test_template_action_create_qa_template_never_mutates_base(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("TEMPLATE_ACTION", "create_qa_template")
    monkeypatch.setenv("QA_TEMPLATE_IMAGE", "ghcr.io/x@sha256:new")
    base = {"id": "07g9dovc17", "name": "EditDNA-Worker-2", "env": {"SECRET": "real-value"}}
    captured = {}

    monkeypatch.setattr(gate, "find_template_by_name", lambda transport, api_key, name: base)

    def _fake_create(transport, api_key, *, base, overrides, log):
        captured["base"] = base
        captured["overrides"] = overrides
        return {"id": "new-tmpl-id", "name": overrides.name, "env": base.get("env", {})}, None

    monkeypatch.setattr(gate, "create_pod_template", _fake_create)

    exit_code = gate.main()

    assert exit_code == 0
    assert captured["base"] is base  # the live-fetched base, never a guessed/hardcoded one
    assert captured["overrides"].name == "CutSell-Pod-QA"
    assert captured["overrides"].image == "ghcr.io/x@sha256:new"
    out = capsys.readouterr().out
    assert "real-value" not in out
    assert _FakeProvider.instances == []


def test_template_action_create_qa_template_aborts_when_base_missing(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("TEMPLATE_ACTION", "create_qa_template")
    monkeypatch.setenv("QA_TEMPLATE_IMAGE", "ghcr.io/x@sha256:new")
    monkeypatch.setattr(gate, "find_template_by_name", lambda transport, api_key, name: None)

    called = {"create": False}
    monkeypatch.setattr(gate, "create_pod_template", lambda *a, **k: called.update(create=True) or (None, "x"))

    exit_code = gate.main()

    assert exit_code == 1
    assert called["create"] is False  # never guesses a payload when the base can't be read


def test_template_action_create_qa_template_rejects_invalid_env_overrides_json(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setenv("TEMPLATE_ACTION", "create_qa_template")
    monkeypatch.setenv("QA_TEMPLATE_IMAGE", "ghcr.io/x@sha256:new")
    monkeypatch.setenv("QA_TEMPLATE_ENV_OVERRIDES_JSON", "not-json")
    monkeypatch.setattr(gate, "find_template_by_name", lambda transport, api_key, name: {"id": "x", "env": {}})

    exit_code = gate.main()

    assert exit_code == 1
    assert _FakeProvider.instances == []
