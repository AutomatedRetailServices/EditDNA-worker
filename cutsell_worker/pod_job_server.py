"""RunPod Pod on-demand transport adapter (D-042: CutSell QA GPU execution
fallback -- RunPod Pod On-Demand automation).

This is the Pod-side counterpart to `serverless_handler.handler()`: the
ONLY Pod-transport-specific code in this module. The canonical CutSell
pipeline is invoked through the exact same
`serverless_handler.run_op(op, payload)` dispatcher RunPod Serverless
uses -- no forked/duplicated business logic between backends.

Deliberately stdlib-only (`http.server`), matching this repo's existing
policy of no new third-party dependency for orchestration/transport code
(see `runpod_orchestration.py`). A CutSell QA Pod runs this file as its
container start command; the orchestrator side (`runpod_pod_provider.py`)
talks to it over plain HTTP on the Pod's exposed port.

Routes:
  GET  /health  -> run_op("health", {})
  POST /run     -> body is {"op": ..., ...payload}; runs run_op(op, body)

Every request is independently guarded: an exception raised inside
`run_op` becomes a well-formed `{"ok": False, "error": ...}` JSON response
(HTTP 500), never an unhandled server crash -- so the orchestrator's health
gate always gets a real, parseable answer instead of a connection reset.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .serverless_handler import run_op

DEFAULT_PORT = int(os.environ.get("CUTSELL_POD_JOB_SERVER_PORT", "8080"))


class _Handler(BaseHTTPRequestHandler):
    server_version = "CutSellPodJobServer/1.0"

    def _write_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _dispatch(self, op: str, payload: dict) -> None:
        try:
            result = run_op(op, payload)
            self._write_json(200, result)
        except Exception as exc:  # noqa: BLE001 -- must never crash the server
            self._write_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:  # noqa: N802 -- stdlib method name
        if self.path.rstrip("/") in ("", "/health"):
            self._dispatch("health", {})
        else:
            self._write_json(404, {"ok": False, "error": f"not found: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 -- stdlib method name
        if self.path.rstrip("/") != "/run":
            self._write_json(404, {"ok": False, "error": f"not found: {self.path}"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError as exc:
            self._write_json(400, {"ok": False, "error": f"invalid JSON body: {exc}"})
            return
        if not isinstance(payload, dict):
            self._write_json(400, {"ok": False, "error": "JSON body must be an object"})
            return
        self._dispatch(str(payload.get("op") or "health"), payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 -- stdlib signature
        # Route through print(..., flush=True) so lines are visible in the
        # Pod's captured stdout/stderr, the same way runpod.serverless's own
        # worker-loop logs are -- never silence request logging outright.
        print(f"[cutsell-pod-job-server] {self.address_string()} {format % args}", flush=True)


def serve(port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    print(f"[cutsell-pod-job-server] listening on 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
