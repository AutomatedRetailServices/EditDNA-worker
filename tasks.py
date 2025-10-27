# /workspace/editdna/tasks.py
# Exposes a stable entrypoint "tasks.job_render" for RQ.
# Uses absolute imports and tolerates both layouts:
#   - worker/jobs.py  (preferred)
#   - jobs.py         (fallback)

from __future__ import annotations
import os, json
from typing import Any, Dict, Optional

# ---- wire up underlying implementation ----
_impl_job_render = None
_impl_run_pipeline = None

try:
    # Prefer namespaced worker package if present
    from worker.jobs import job_render as _impl_job_render  # type: ignore
    try:
        from worker.jobs import run_pipeline as _impl_run_pipeline  # type: ignore
    except Exception:
        _impl_run_pipeline = None
except Exception:
    # Fallback to flat layout
    try:
        from jobs import job_render as _impl_job_render  # type: ignore
        try:
            from jobs import run_pipeline as _impl_run_pipeline  # type: ignore
        except Exception:
            _impl_run_pipeline = None
    except Exception as e:
        raise ImportError(
            "Could not import job implementation from worker.jobs or jobs. "
            "Make sure one of these files defines job_render (and optionally run_pipeline)."
        ) from e


def _coerce_payload(payload: Any) -> Dict:
    """Accept dict (preferred), JSON string, or None."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (str, bytes)):
        try:
            return json.loads(payload)
        except Exception:
            # last resort – pass it through so we don't crash RQ
            return {"raw_payload": payload if isinstance(payload, str) else payload.decode("utf-8", "ignore")}
    return {}


def _apply_option_env_overrides(options: Dict) -> None:
    """Allow /render payload.options to override env at runtime (strings only)."""
    for k, v in (options or {}).items():
        try:
            os.environ[str(k)] = str(v)
        except Exception:
            pass


def job_render(payload: Any) -> Dict:
    """
    Stable entrypoint called by RQ: enqueue as 'tasks.job_render'.
    Accepts the exact payload you send from the web/API.
    """
    p = _coerce_payload(payload)

    # optional local path hint (when web service downloaded the file already)
    local_path = p.get("local_path") or p.get("input_local")

    # allow runtime option overrides (FUNNEL_COUNTS, MAX_DURATION_SEC, etc.)
    _apply_option_env_overrides(p.get("options") or {})

    # If your real code provides run_pipeline(local_path=..., payload=...), prefer it.
    if _impl_run_pipeline is not None:
        return _impl_run_pipeline(local_path=local_path, payload=p)

    # Otherwise call legacy signature job_render(local_path)
    return _impl_job_render(local_path)
