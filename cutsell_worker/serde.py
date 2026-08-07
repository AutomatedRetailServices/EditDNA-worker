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
    TextOverlay,
    Word,
)

CAPTION_PRESETS = {"classic", "clean"}


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


def _word_from_dict(item: dict) -> Word:
    return Word(
        text=str(item.get("text") or ""),
        start=float(item["start"]),
        end=float(item["end"]),
        confidence=(float(item["confidence"]) if item.get("confidence") is not None else None),
    )


def _draft_clip_from_dict(item: dict, *, selected_default: bool) -> DraftClip:
    words = tuple(_word_from_dict(dict(word)) for word in item.get("words") or ())
    volume = float(item.get("audio_volume", 1.0))
    if volume < 0.0 or volume > 2.0:
        raise ValueError("audio_volume must be between 0.0 and 2.0")
    return DraftClip(
        clip_id=str(item["clip_id"]),
        source_asset_id=str(item["source_asset_id"]),
        source_order=int(item.get("source_order", 0)),
        start=float(item["start"]),
        end=float(item["end"]),
        text=str(item.get("text") or ""),
        caption_text=str(item.get("caption_text") if item.get("caption_text") is not None else item.get("text") or ""),
        words=words,
        semantic_role=SemanticRole(str(item.get("semantic_role") or SemanticRole.OTHER.value)),
        take_group_id=(str(item["take_group_id"]) if item.get("take_group_id") else None),
        selected=bool(item.get("selected", selected_default)),
        audio_muted=bool(item.get("audio_muted", False)),
        audio_volume=volume,
    )


def _text_overlay_from_dict(item: dict) -> TextOverlay:
    text = str(item.get("text") or "").strip()
    if not text or len(text) > 500:
        raise ValueError("text overlay text must contain 1 to 500 characters")
    start = float(item["start"])
    end = float(item["end"])
    x = float(item.get("x", 0.5))
    y = float(item.get("y", 0.2))
    scale = float(item.get("scale", 1.0))
    if start < 0 or end <= start:
        raise ValueError("text overlay end must be after start")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("text overlay position must be normalized 0 to 1")
    if not 0.5 <= scale <= 3.0:
        raise ValueError("text overlay scale must be between 0.5 and 3.0")
    return TextOverlay(
        overlay_id=str(item["overlay_id"]),
        text=text,
        start=start,
        end=end,
        x=x,
        y=y,
        scale=scale,
    )


def draft_from_dict(payload: dict) -> DraftTimeline:
    if not isinstance(payload, dict):
        raise ValueError("draft must be an object")
    selected = tuple(_draft_clip_from_dict(dict(item), selected_default=True) for item in payload.get("selected") or ())
    alternates = tuple(_draft_clip_from_dict(dict(item), selected_default=False) for item in payload.get("alternates") or ())
    discarded = tuple(_draft_clip_from_dict(dict(item), selected_default=False) for item in payload.get("discarded") or ())
    if not selected:
        raise ValueError("draft requires at least one selected clip")
    preset = str(payload.get("caption_preset") or "classic")
    if preset not in CAPTION_PRESETS:
        raise ValueError("caption_preset must be classic or clean")
    overlays = tuple(_text_overlay_from_dict(dict(item)) for item in payload.get("text_overlays") or ())
    total_duration = sum(max(0.0, clip.end - clip.start) for clip in selected)
    if any(overlay.end > total_duration + 1e-6 for overlay in overlays):
        raise ValueError("text overlay exceeds draft timeline duration")
    return DraftTimeline(
        schema_version=str(payload.get("schema_version") or "cutsell.v1"),
        project_id=str(payload["project_id"]),
        strategy=EditStrategy(str(payload.get("strategy") or EditStrategy.MIXED.value)),
        selected=selected,
        alternates=alternates,
        discarded=discarded,
        diagnostics=dict(payload.get("diagnostics") or {}),
        captions_enabled=bool(payload.get("captions_enabled", True)),
        caption_preset=preset,
        text_overlays=overlays,
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
