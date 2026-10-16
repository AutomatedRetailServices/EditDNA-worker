"""Editor v5 "Clean up voice" (build 17): ask for / follow the cleaned copy of a sound.

    POST /v1/projects/{project_id}/clean-audio   {user_id, target: source|asset, id}
         -> ready (with a short-lived link) | cleaning (job started now or already running)
    GET  /v1/projects/{project_id}/clean-audio?user_id=&target=&id=
         -> ready | cleaning (progress 0-100) | failed | missing

`user_id` is checked against the signed-in user by AuthScopeMiddleware (body and query), and again
here. The client names a source or a timeline asset by id only; the worker resolves the stored
file through the same tenant checks as the export (audio_clean.fetch_input).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from cutsell_worker import audio_clean, storage_routing
from cutsell_worker.config import load_runtime_config

router = APIRouter()
LINK_SECONDS = 3600
#: A job that has said nothing for this long is treated as lost and may be started again.
STALE_SEC = 15 * 60


class CleanAudioRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=200)
    target: Literal["source", "asset"]
    id: str = Field(min_length=1, max_length=200)


def _owner(request: Request, user_id: str) -> None:
    auth_user_id = getattr(request.state, "auth_user_id", None)
    if auth_user_id is not None and str(auth_user_id) != str(user_id):
        raise HTTPException(status_code=403, detail="project does not belong to this user")


def _redis():
    config = load_runtime_config()
    if not config.redis_url:
        raise HTTPException(status_code=503, detail="REDIS_URL is not configured")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _ready(user_id: str, project_id: str, target: str, ident: str) -> dict | None:
    bucket = load_runtime_config().s3_bucket
    if not bucket:
        raise HTTPException(status_code=503, detail="storage is not configured")
    key = audio_clean.clean_key(user_id=user_id, project_id=project_id, target=target, ident=ident)
    try:
        storage_routing.head(bucket, key)
    except Exception as exc:
        if storage_routing.is_not_found(exc):
            return None
        raise HTTPException(status_code=503, detail="storage is unavailable") from None
    url = storage_routing.presigned_get(bucket, key, expires_in=LINK_SECONDS)
    return {"status": "ready", "progress": 100, "url": url, "expires_in": LINK_SECONDS}


def _status(user_id: str, project_id: str, target: str, ident: str, redis) -> dict:
    ready = _ready(user_id, project_id, target, ident)
    if ready:
        return ready
    record = audio_clean.read_status(redis, user_id=user_id, project_id=project_id, target=target, ident=ident) or {}
    state = str(record.get("state") or "missing")
    if state in ("queued", "cleaning"):
        import time
        if time.time() - float(record.get("updated_at") or record.get("queued_at") or 0) > STALE_SEC:
            state = "missing"
    if state == "ready":
        state = "missing"  # the record says ready but the file is gone: clean again
    out = {"status": state if state in ("queued", "cleaning", "failed") else "missing",
           "progress": int(record.get("progress") or 0)}
    if out["status"] == "queued":
        out["status"] = "cleaning"
    if out["status"] == "failed":
        out["error"] = str(record.get("error") or "clean_audio_failed")[:200]
    return out


@router.post("/v1/projects/{project_id}/clean-audio")
def start_clean_audio(project_id: str, payload: CleanAudioRequest, request: Request):
    _owner(request, payload.user_id)
    redis = _redis()
    if payload.target == "source":
        from cutsell_worker.draft_store import get_draft_snapshot
        try:
            snapshot = get_draft_snapshot(user_id=payload.user_id, project_id=project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="draft not found") from None
        if not audio_clean.source_uri(snapshot, payload.id):
            raise HTTPException(status_code=404, detail="source not found in this project")
    current = _status(payload.user_id, project_id, payload.target, payload.id, redis)
    if current["status"] in ("ready", "cleaning"):
        return current
    from cutsell_worker.queueing import enqueue_clean_audio
    import time
    audio_clean.write_status(redis, user_id=payload.user_id, project_id=project_id, target=payload.target,
                             ident=payload.id, state="queued", progress=0, queued_at=time.time(),
                             updated_at=time.time(), error=None)
    submission = enqueue_clean_audio({"user_id": payload.user_id, "project_id": project_id,
                                      "target": payload.target, "id": payload.id})
    return {"status": "cleaning", "progress": 0, "job_id": submission.job_id}


@router.get("/v1/projects/{project_id}/clean-audio")
def clean_audio_status(project_id: str, user_id: str, target: Literal["source", "asset"], id: str,
                       request: Request):
    _owner(request, user_id)
    if not id or len(id) > 200:
        raise HTTPException(status_code=422, detail="id is required")
    return _status(user_id, project_id, target, id, _redis())
