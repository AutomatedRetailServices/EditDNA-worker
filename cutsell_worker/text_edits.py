"""Stateless V1 text-overlay edits on the final Draft timeline."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping


def _draft_duration(draft: Mapping[str, Any]) -> float:
    total = 0.0
    selected = draft.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("draft requires selected clips")
    for clip in selected:
        total += max(0.0, float(clip["end"]) - float(clip["start"]))
    return total


def _validate(*, text: str, start: float, end: float, x: float, y: float, scale: float, duration: float) -> None:
    if not str(text).strip() or len(str(text)) > 500:
        raise ValueError("text overlay text must contain 1 to 500 characters")
    if start < 0 or end <= start or end > duration + 1e-6:
        raise ValueError("text overlay timing must stay inside draft duration")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("text overlay position must be normalized 0 to 1")
    if not 0.5 <= scale <= 3.0:
        raise ValueError("text overlay scale must be between 0.5 and 3.0")


def _id(project_id: str, text: str, start: float, end: float, index: int) -> str:
    digest = hashlib.sha256(f"{project_id}|{text}|{start:.3f}|{end:.3f}|{index}".encode()).hexdigest()[:16]
    return f"txt_{digest}"


def add_text_overlay(
    draft: Mapping[str, Any], *, text: str, start: float, end: float,
    x: float = 0.5, y: float = 0.2, scale: float = 1.0,
) -> dict[str, Any]:
    out = deepcopy(dict(draft))
    overlays = list(out.get("text_overlays") or [])
    duration = _draft_duration(out)
    resolved = (str(text), float(start), float(end), float(x), float(y), float(scale))
    _validate(text=resolved[0], start=resolved[1], end=resolved[2], x=resolved[3], y=resolved[4], scale=resolved[5], duration=duration)
    overlay = {
        "overlay_id": _id(str(out.get("project_id") or ""), resolved[0], resolved[1], resolved[2], len(overlays)),
        "text": resolved[0].strip(),
        "start": round(resolved[1], 3),
        "end": round(resolved[2], 3),
        "x": round(resolved[3], 4),
        "y": round(resolved[4], 4),
        "scale": round(resolved[5], 3),
    }
    overlays.append(overlay)
    out["text_overlays"] = overlays
    return out


def update_text_overlay(draft: Mapping[str, Any], overlay_id: str, **changes) -> dict[str, Any]:
    out = deepcopy(dict(draft))
    overlays = list(out.get("text_overlays") or [])
    position = next((i for i, item in enumerate(overlays) if str(item.get("overlay_id")) == overlay_id), None)
    if position is None:
        raise ValueError("text overlay not found")
    item = dict(overlays[position])
    for field in ("text", "start", "end", "x", "y", "scale"):
        if changes.get(field) is not None:
            item[field] = changes[field]
    duration = _draft_duration(out)
    _validate(
        text=str(item.get("text") or ""), start=float(item["start"]), end=float(item["end"]),
        x=float(item.get("x", 0.5)), y=float(item.get("y", 0.2)), scale=float(item.get("scale", 1.0)), duration=duration,
    )
    item.update({
        "text": str(item["text"]).strip(), "start": round(float(item["start"]), 3), "end": round(float(item["end"]), 3),
        "x": round(float(item.get("x", 0.5)), 4), "y": round(float(item.get("y", 0.2)), 4), "scale": round(float(item.get("scale", 1.0)), 3),
    })
    overlays[position] = item
    out["text_overlays"] = overlays
    return out


def remove_text_overlay(draft: Mapping[str, Any], overlay_id: str) -> dict[str, Any]:
    out = deepcopy(dict(draft))
    overlays = list(out.get("text_overlays") or [])
    kept = [item for item in overlays if str(item.get("overlay_id")) != overlay_id]
    if len(kept) == len(overlays):
        raise ValueError("text overlay not found")
    out["text_overlays"] = kept
    return out
