"""JSON-safe serialization helpers for RQ/API boundaries."""
from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import Any

from .contracts import ProcessingRequest, SourceAsset


def request_from_dict(payload: dict) -> ProcessingRequest:
    sources = tuple(
        SourceAsset(
            source_asset_id=str(item["source_asset_id"]),
            project_id=str(payload["project_id"]),
            user_id=str(payload["user_id"]),
            original_name=str(item.get("original_name") or "source"),
            source_order=int(item.get("source_order", index)),
            duration_sec=float(item.get("duration_sec") or 0.0),
            uri=str(item["uri"]),
            has_audio=bool(item.get("has_audio", True)),
            metadata=dict(item.get("metadata") or {}),
        )
        for index, item in enumerate(payload.get("sources") or ())
    )
    if not sources:
        raise ValueError("processing request requires at least one source")
    return ProcessingRequest(
        project_id=str(payload["project_id"]),
        user_id=str(payload["user_id"]),
        sources=sources,
        preferred_source_order=tuple(payload.get("preferred_source_order") or ()),
        audio_overlap=bool(payload.get("audio_overlap", False)),
        language_hint=(str(payload["language_hint"]) if payload.get("language_hint") else None),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return value


def result_to_dict(result) -> dict:
    return _json_safe(result)
