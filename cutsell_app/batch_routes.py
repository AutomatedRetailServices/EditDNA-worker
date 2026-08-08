"""Batch API for high-output CutSell creators."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from cutsell_worker.batch import create_batch, get_batch
from cutsell_worker.queueing import enqueue_batch_item
from cutsell_worker.source_identity import stable_source_id
from cutsell_worker.uploads import validate_product_source_uri

router = APIRouter(prefix="/v1/batches", tags=["batch"])


class BatchSource(BaseModel):
    original_name: str
    uri: str
    source_order: int = Field(ge=0)
    duration_sec: float = Field(default=0.0, ge=0.0)


class BatchProject(BaseModel):
    project_id: str
    sources: list[BatchSource] = Field(min_length=1)
    language_hint: str | None = None
    audio_overlap: bool = False


class BatchCreateRequest(BaseModel):
    user_id: str
    projects: list[BatchProject] = Field(min_length=1, max_length=10)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_batch_job(payload: BatchCreateRequest):
    try:
        project_ids = [item.project_id for item in payload.projects]
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("batch project IDs must be unique")
        processing_payloads = []
        for project in payload.projects:
            seen_orders = set()
            sources = []
            for source in project.sources:
                if source.source_order in seen_orders:
                    raise ValueError("source_order values must be unique within each project")
                seen_orders.add(source.source_order)
                validate_product_source_uri(
                    source.uri,
                    user_id=payload.user_id,
                    project_id=project.project_id,
                )
                sources.append({
                    "source_asset_id": stable_source_id(
                        project.project_id, source.source_order, source.original_name
                    ),
                    "original_name": source.original_name,
                    "source_order": source.source_order,
                    "duration_sec": source.duration_sec,
                    "uri": source.uri,
                })
            processing_payloads.append({
                "project_id": project.project_id,
                "user_id": payload.user_id,
                "sources": sources,
                "language_hint": project.language_hint,
                "audio_overlap": project.audio_overlap,
            })
        record = create_batch(user_id=payload.user_id, payloads=processing_payloads)
        submission = enqueue_batch_item(
            batch_id=record["batch_id"], user_id=payload.user_id, index=0
        )
        return {
            "batch_id": record["batch_id"],
            "state": "queued",
            "item_count": len(processing_payloads),
            "first_job_id": submission.job_id,
            "queue": submission.queue_name,
        }
    except PermissionError:
        raise HTTPException(status_code=403, detail="batch ownership mismatch") from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("/{batch_id}")
def get_batch_status(batch_id: str, user_id: str):
    try:
        return get_batch(user_id=user_id, batch_id=batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="batch not found") from None
    except PermissionError:
        raise HTTPException(status_code=403, detail="batch does not belong to this user") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
