"""Provider-neutral dispatch contract (RunPod Pod On-Demand execution-fallback
work): `serverless_handler.run_op(op, payload)` must be the ONE dispatch
table both the RunPod Serverless `handler()` adapter and the future RunPod
Pod job server call -- never a forked/duplicated implementation.

These tests lock: (1) `handler()` delegates to `run_op()` rather than
re-implementing dispatch, (2) `run_op()` alone (no RunPod job envelope)
produces identical output to `handler()` for every op, (3) op
normalization (case/whitespace, default to "health") behaves identically
through both entrypoints, (4) an unsupported op raises the same error
through both entrypoints.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest


def _stub_missing_module(name, **attributes):
    if name in sys.modules:
        return
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module


_stub_missing_module("runpod", serverless=types.SimpleNamespace(start=lambda *_a, **_k: None))

from cutsell_worker import serverless_handler  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_health(monkeypatch):
    # _health() imports real torch, which is not installed in this dev
    # sandbox -- stub it to a deterministic sentinel so this test suite
    # only exercises dispatch, not hardware probing (already covered by
    # tests/test_cutsell_serverless_health_diagnostics.py).
    sentinel = {"ok": True, "cuda_available": False}
    monkeypatch.setattr(serverless_handler, "_health", lambda: dict(sentinel))
    return sentinel


def test_handler_delegates_to_run_op_for_health(_stub_health):
    via_handler = serverless_handler.handler({"input": {"op": "health"}})
    via_run_op = serverless_handler.run_op("health", {})
    assert via_handler == via_run_op == _stub_health


def test_handler_default_op_is_health_through_run_op(_stub_health):
    assert serverless_handler.handler({"input": {}}) == _stub_health
    assert serverless_handler.handler({}) == _stub_health
    assert serverless_handler.run_op("", {}) == _stub_health


def test_handler_and_run_op_normalize_case_and_whitespace_identically(_stub_health):
    assert serverless_handler.handler({"input": {"op": " HEALTH "}}) == _stub_health
    assert serverless_handler.run_op(" HEALTH ", {}) == _stub_health


def test_handler_and_run_op_reject_unsupported_op_identically():
    with pytest.raises(ValueError, match="unsupported op: bogus"):
        serverless_handler.handler({"input": {"op": "bogus"}})
    with pytest.raises(ValueError, match="unsupported op: bogus"):
        serverless_handler.run_op("bogus", {})


def test_handler_passes_full_payload_through_to_run_op(monkeypatch):
    seen = {}

    def _fake_run_op(op, payload):
        seen["op"] = op
        seen["payload"] = payload
        return {"ok": True, "echo": payload}

    monkeypatch.setattr(serverless_handler, "run_op", _fake_run_op)
    job = {"input": {"op": "focused", "source_key": "s3://x", "benchmark_id": "b1"}}
    result = serverless_handler.handler(job)
    assert seen["op"] == "focused"
    assert seen["payload"] == job["input"]
    assert result == {"ok": True, "echo": job["input"]}
