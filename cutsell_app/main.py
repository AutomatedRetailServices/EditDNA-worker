"""Standalone FastAPI entrypoint for the clean CutSell mobile backend."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from cutsell_worker.config import load_runtime_config
from cutsell_worker.queueing import enqueue_flow_b
from cutsell_worker.source_identity import stable_source_id

app = FastAPI(title="CutSell API", version="0.1.0")


class SourceInput(BaseModel):
    original_name: str
    uri: str
    source_order: int = Field(ge=0)
    duration_sec: float = Field(default=0.0, ge=0.0)


class FlowBSubmitRequest(BaseModel):
    project_id: str
    user_id: str
    sources: list[SourceInput] = Field(min_length=1)
    language_hint: str | None = None
    audio_overlap: bool = False


class FlowBSubmitResponse(BaseModel):
    job_id: str
    queue: str
    state: str = "uploaded"


@app.get("/v1/healthz")
def healthz():
    config = load_runtime_config()
    return {
        "ok": True,
        "service": "cutsell-api",
        "version": "0.1.0",
        "queue_ready": config.queue_ready,
        "storage_ready": config.storage_ready,
        "semantic_ready": config.semantic_ready,
    }


@app.post("/v1/flow-b/jobs", response_model=FlowBSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_flow_b(payload: FlowBSubmitRequest):
    sources = []
    seen_orders = set()
    for item in payload.sources:
        if item.source_order in seen_orders:
            raise HTTPException(status_code=409, detail="source_order values must be unique")
        seen_orders.add(item.source_order)
        sources.append({
            "source_asset_id": stable_source_id(payload.project_id, item.source_order, item.original_name),
            "original_name": item.original_name,
            "source_order": item.source_order,
            "duration_sec": item.duration_sec,
            "uri": item.uri,
        })
    submission = enqueue_flow_b({
        "project_id": payload.project_id,
        "user_id": payload.user_id,
        "sources": sources,
        "language_hint": payload.language_hint,
        "audio_overlap": payload.audio_overlap,
    })
    return FlowBSubmitResponse(job_id=submission.job_id, queue=submission.queue_name)
