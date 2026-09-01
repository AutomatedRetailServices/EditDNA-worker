"""Immutable production feedback events for CutSell evaluation and learning."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from .config import load_runtime_config

FEEDBACK_PREFIX = "cutsell/feedback/"
ALLOWED_RATINGS = {"good", "bad", "report"}


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("feedback scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def build_feedback_event(
    *,
    user_id: str,
    project_id: str,
    rating: str,
    draft: dict,
    reason: str | None = None,
    clip_id: str | None = None,
    time_sec: float | None = None,
    processing_metrics: dict | None = None,
) -> dict:
    normalized_rating = str(rating or "").strip().lower()
    if normalized_rating not in ALLOWED_RATINGS:
        raise ValueError("feedback rating must be good, bad, or report")
    if str(draft.get("project_id") or "") != project_id:
        raise ValueError("feedback draft project does not match project")
    selected = list(draft.get("selected") or ())
    alternates = list(draft.get("alternates") or ())
    if not selected:
        raise ValueError("feedback requires a finished draft with selected clips")
    known_clip_ids = {
        str(item.get("clip_id") or "")
        for item in selected + alternates + list(draft.get("discarded") or ())
        if isinstance(item, dict)
    }
    if clip_id and clip_id not in known_clip_ids:
        raise ValueError("feedback clip marker is not part of this draft")
    marker_time = None if time_sec is None else float(time_sec)
    if marker_time is not None and marker_time < 0:
        raise ValueError("feedback time marker must be non-negative")
    config = load_runtime_config()
    diagnostics = dict(draft.get("diagnostics") or {})
    return {
        "schema_version": "cutsell.feedback.v1",
        "feedback_id": f"fb_{uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "user_scope": _scope_hash(user_id),
        "rating": normalized_rating,
        "reason": (str(reason).strip()[:500] if reason else None),
        "marker": {
            "clip_id": clip_id,
            "time_sec": marker_time,
        } if clip_id is not None or marker_time is not None else None,
        "strategy": str(draft.get("strategy") or "mixed"),
        "models": {
            "asr": config.asr_model,
            "semantic": config.semantic_model,
            "visual": config.visual_model,
            "take_judge": config.take_judge_model,
        },
        "selected": [
            {
                "clip_id": item.get("clip_id"),
                "source_asset_id": item.get("source_asset_id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "semantic_role": item.get("semantic_role"),
                "take_group_id": item.get("take_group_id"),
            }
            for item in selected if isinstance(item, dict)
        ],
        "alternates": [
            {
                "clip_id": item.get("clip_id"),
                "source_asset_id": item.get("source_asset_id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "semantic_role": item.get("semantic_role"),
                "take_group_id": item.get("take_group_id"),
            }
            for item in alternates if isinstance(item, dict)
        ],
        "execution": {
            "take_judge_status_counts": diagnostics.get("take_judge_status_counts"),
            "take_judge_fallback_reasons": diagnostics.get("take_judge_fallback_reasons"),
            "semantic_status": diagnostics.get("semantic_status"),
            "visual_status": diagnostics.get("visual_status"),
        },
        "processing_metrics": dict(processing_metrics or {}),
    }


def store_feedback_event(event: dict, *, user_id: str, project_id: str, client=None) -> dict:
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required for feedback storage")
    if str(event.get("project_id") or "") != project_id:
        raise ValueError("feedback event project mismatch")
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    feedback_id = str(event.get("feedback_id") or "")
    if not feedback_id.startswith("fb_"):
        raise ValueError("feedback event ID is invalid")
    key = (
        f"{FEEDBACK_PREFIX}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"
        f"{feedback_id}.json"
    )
    body = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    client.put_object(
        Bucket=config.s3_bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return {
        "feedback_id": feedback_id,
        "feedback_uri": f"s3://{config.s3_bucket}/{key}",
        "stored": True,
    }
