"""RunPod Pod on-demand transport adapter tests (D-042). Spins up the real
stdlib HTTP server on an ephemeral port and drives it over real HTTP --
no fakes for the transport layer itself, since the whole point of this
module is that it IS the transport. `run_op` is monkeypatched so these
tests exercise only the HTTP adapter, not the editorial pipeline (already
covered elsewhere).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import types
import urllib.error
import urllib.request

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

from cutsell_worker import pod_job_server  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402


@pytest.fixture
def running_server(monkeypatch):
    calls = []

    def _fake_run_op(op, payload):
        calls.append((op, payload))
        if op == "boom":
            raise RuntimeError("simulated failure")
        return {"ok": True, "op": op, "payload": payload}

    monkeypatch.setattr(pod_job_server, "run_op", _fake_run_op)
    server = ThreadingHTTPServer(("127.0.0.1", 0), pod_job_server._Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _get_expect_error(url: str):
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, body: dict | bytes):
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_get_health_dispatches_health_op(running_server):
    base, calls = running_server
    status, body = _get(f"{base}/health")
    assert status == 200
    assert body == {"ok": True, "op": "health", "payload": {}}
    assert calls == [("health", {})]


def test_get_root_also_dispatches_health(running_server):
    base, _calls = running_server
    status, body = _get(base)
    assert status == 200
    assert body["op"] == "health"


def test_get_unknown_path_is_404(running_server):
    base, _calls = running_server
    status, body = _get_expect_error(f"{base}/nope")
    assert status == 404
    assert body["ok"] is False


def test_post_run_dispatches_with_full_payload(running_server):
    base, calls = running_server
    payload = {"op": "focused", "source_key": "s3://x", "benchmark_id": "b1"}
    status, body = _post(f"{base}/run", payload)
    assert status == 200
    assert body == {"ok": True, "op": "focused", "payload": payload}
    assert calls == [("focused", payload)]


def test_post_run_missing_op_defaults_to_health(running_server):
    base, calls = running_server
    status, body = _post(f"{base}/run", {})
    assert status == 200
    assert calls == [("health", {})]


def test_post_run_invalid_json_is_400(running_server):
    base, _calls = running_server
    status, body = _post(f"{base}/run", b"not json")
    assert status == 400
    assert body["ok"] is False


def test_post_run_non_object_json_is_400(running_server):
    base, _calls = running_server
    status, body = _post(f"{base}/run", b"[1,2,3]")
    assert status == 400
    assert body["ok"] is False


def test_post_unknown_path_is_404(running_server):
    base, _calls = running_server
    status, body = _post(f"{base}/other", {"op": "health"})
    assert status == 404


def test_run_op_exception_becomes_500_not_a_crash(running_server):
    base, calls = running_server
    status, body = _post(f"{base}/run", {"op": "boom"})
    assert status == 500
    assert body["ok"] is False
    assert "simulated failure" in body["error"]
    # The server must still be alive and answer the next request normally.
    status2, body2 = _get(f"{base}/health")
    assert status2 == 200
    assert body2["ok"] is True
