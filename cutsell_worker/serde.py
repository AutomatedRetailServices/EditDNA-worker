"""JSON-safe serialization helpers for RQ/API boundaries."""
from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import Any

from .contracts import (
    DraftClip,
    DraftTimeline,
    EditStrategy,
    ProcessingRequest,
    SemanticRole,
    SourceAsset,
)


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


def _draft_clip_from_dict(item: dict, *, selected_default: bool) -> DraftClip:
    return DraftClip(
        clip_id=str(item["clip_id"]),
        source_asset_id=str(item["source_asset_id"]),
        source_order=int(item.get("source_order", 0)),
        start=float(item["start"]),
        end=float(item["end"]),
        text=str(item.get("text") or ""),
        caption_text=str(item.get("caption_text") if item.get("caption_text") is not None else item.get("text") or ""),
        semantic_role=SemanticRole(str(item.get("semantic_role") or SemanticRole.OTHER.value)),
        take_group_id=(str(item["take_group_id"]) if item.get("take_group_id") else None),
        selected=bool(item.get("selected", selected_default)),
    )


def draft_from_dict(payload: dict) -> DraftTimeline:
    if not isinstance(payload, dict):
        raise ValueError("draft must be an object")
    selected = tuple(_draft_clip_from_dict(dict(item), selected_default=True) for item in payload.get("selected") or ())
    alternates = tuple(_draft_clip_from_dict(dict(item), selected_default=False) for item in payload.get("alternates") or ())
    discarded = tuple(_draft_clip_from_dict(dict(item), selected_default=False) for item in payload.get("discarded") or ())
    if not selected:
        raise ValueError("draft requires at least one selected clip")
    return DraftTimeline(
        schema_version=str(payload.get("schema_version") or "cutsell.v1"),
        project_id=str(payload["project_id"]),
        strategy=EditStrategy(str(payload.get("strategy") or EditStrategy.MIXED.value)),
        selected=selected,
        alternates=alternates,
        discarded=discarded,
        diagnostics=dict(payload.get("diagnostics") or {}),
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
