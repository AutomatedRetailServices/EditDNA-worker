"""Stateless photo/video overlay edits for the CutSell final timeline."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping


def _duration(draft: Mapping[str, Any]) -> float:
    selected = draft.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("draft requires selected clips")
    return sum(max(0.0, float(item["end"]) - float(item["start"])) for item in selected)


def _validate(
    *, kind: str, uri: str, start: float, end: float,
    x: float, y: float, width: float, source_start: float,
    source_end: float | None, duration: float,
) -> None:
    if kind not in {"photo", "video"}:
        raise ValueError("overlay kind must be photo or video")
    if not uri:
        raise ValueError("overlay uri is required")
    if start < 0 or end <= start or end > duration + 1e-6:
        raise ValueError("overlay timing must stay inside draft duration")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("overlay position must be normalized 0 to 1")
    if not 0.1 <= width <= 1.0:
        raise ValueError("overlay width must be between 0.1 and 1.0")
    if source_start < 0 or (source_end is not None and source_end <= source_start):
        raise ValueError("overlay source trim is invalid")
    if kind == "photo" and (source_start != 0.0 or source_end is not None):
        raise ValueError("photo overlay cannot use source trim")


def _id(project_id: str, uri: str, start: float, end: float, index: int) -> str:
    digest = hashlib.sha256(f"{project_id}|{uri}|{start:.3f}|{end:.3f}|{index}".encode()).hexdigest()[:16]
    return f"ovl_{digest}"


def add_media_overlay(
    draft: Mapping[str, Any], *, kind: str, uri: str, start: float, end: float,
    x: float = 0.5, y: float = 0.5, width: float = 0.4,
    source_start: float = 0.0, source_end: float | None = None,
    mute_audio: bool = True,
) -> dict[str, Any]:
    out = deepcopy(dict(draft))
    overlays = list(out.get("media_overlays") or [])
    values = (str(kind), str(uri), float(start), float(end), float(x), float(y), float(width), float(source_start), float(source_end) if source_end is not None else None)
    _validate(kind=values[0], uri=values[1], start=values[2], end=values[3], x=values[4], y=values[5], width=values[6], source_start=values[7], source_end=values[8], duration=_duration(out))
    overlays.append({
        "overlay_id": _id(str(out.get("project_id") or ""), values[1], values[2], values[3], len(overlays)),
        "kind": values[0], "uri": values[1], "start": round(values[2], 3), "end": round(values[3], 3),
        "x": round(values[4], 4), "y": round(values[5], 4), "width": round(values[6], 4),
        "source_start": round(values[7], 3), "source_end": (round(values[8], 3) if values[8] is not None else None),
        "mute_audio": bool(mute_audio),
    })
    out["media_overlays"] = overlays
    return out


def update_media_overlay(draft: Mapping[str, Any], overlay_id: str, **changes) -> dict[str, Any]:
    out = deepcopy(dict(draft))
    overlays = list(out.get("media_overlays") or [])
    position = next((i for i, item in enumerate(overlays) if str(item.get("overlay_id")) == overlay_id), None)
    if position is None:
        raise ValueError("media overlay not found")
    item = dict(overlays[position])
    for field in ("start", "end", "x", "y", "width", "source_start", "source_end", "mute_audio"):
        if field in changes and changes[field] is not None:
            item[field] = changes[field]
    source_end = float(item["source_end"]) if item.get("source_end") is not None else None
    _validate(
        kind=str(item["kind"]), uri=str(item["uri"]), start=float(item["start"]), end=float(item["end"]),
        x=float(item.get("x", 0.5)), y=float(item.get("y", 0.5)), width=float(item.get("width", 0.4)),
        source_start=float(item.get("source_start", 0.0)), source_end=source_end, duration=_duration(out),
    )
    item.update({
        "start": round(float(item["start"]), 3), "end": round(float(item["end"]), 3),
        "x": round(float(item.get("x", 0.5)), 4), "y": round(float(item.get("y", 0.5)), 4),
        "width": round(float(item.get("width", 0.4)), 4), "source_start": round(float(item.get("source_start", 0.0)), 3),
        "source_end": (round(source_end, 3) if source_end is not None else None), "mute_audio": bool(item.get("mute_audio", True)),
    })
    overlays[position] = item
    out["media_overlays"] = overlays
    return out


def remove_media_overlay(draft: Mapping[str, Any], overlay_id: str) -> dict[str, Any]:
    out = deepcopy(dict(draft))
    overlays = list(out.get("media_overlays") or [])
    kept = [item for item in overlays if str(item.get("overlay_id")) != overlay_id]
    if len(kept) == len(overlays):
        raise ValueError("media overlay not found")
    out["media_overlays"] = kept
    return out
