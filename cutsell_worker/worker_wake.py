"""Tell an on-demand worker (modal_cutsell_worker.py) that a job was just enqueued.

Best effort by design: the job is already safely on the Redis queue when this runs, and the worker's
own periodic sweep picks up anything a failed wake leaves behind. So this never raises and never
delays the API response for long. With no CUTSELL_WORKER_WAKE_URL configured it does nothing, which
is the behavior of every deployment that runs an always-on RQ worker.
"""
from __future__ import annotations

import os

WAKE_TIMEOUT_SEC = 4.0


def wake_worker() -> dict:
    url = os.environ.get("CUTSELL_WORKER_WAKE_URL", "").strip()
    if not url:
        return {"status": "not_configured"}
    token = os.environ.get("CUTSELL_WORKER_WAKE_TOKEN", "").strip()
    if not url.lower().startswith("https://") or not token:
        return {"status": "misconfigured"}
    try:
        import requests

        response = requests.post(url, json={"token": token}, timeout=WAKE_TIMEOUT_SEC)
        if 200 <= response.status_code < 300:
            return {"status": "woken"}
        return {"status": "failed", "http_status": int(response.status_code)}
    except Exception as exc:  # network trouble must never fail an enqueue that already succeeded
        return {"status": "failed", "reason": exc.__class__.__name__}
