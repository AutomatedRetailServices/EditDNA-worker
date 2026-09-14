"""Pure stateless edits for the CutSell mobile Draft Timeline.

Every operation returns a new JSON-safe draft and never mutates its input. The
mobile client can therefore keep its own undo/redo stack. Commercial meaning never
grants deletion authority here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping, Sequence


class DraftEditError(ValueError):
    pass


MIN_TRIM_DURATION_SEC = 0.15
MIN_SPLIT_DURATION_SEC = 0.15


def _clip_id(item: Mapping[str, Any]) -> str:
    return str(item.get("clip_id") or "")


def _group_id(item: Mapping[str, Any]) -> str:
    return str(item.get("take_group_id") or "")


def _index(items: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {_clip_id(item): item for item in items if _clip_id(item)}


def _copy_draft(draft: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(draft))
    for field in ("selected", "alternates", "discarded"):
        if field not in out or not isinstance(out[field], list):
            raise DraftEditError(f"draft requires {field} list")
    return out


def _words(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = item.get("words") or []
    if not isinstance(raw, list):
        raise DraftEditError("clip word timings must be a list")
    output = []
    for word in raw:
        if not isinstance(word, Mapping):
            raise DraftEditError("clip word timing is invalid")
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            raise DraftEditError("clip word timing is invalid") from None
        if end < start:
            raise DraftEditError("clip word timing is invalid")
        output.append({
            "text": str(word.get("text") or ""),
            "start": start,
            "end": end,
            **({"confidence": word.get("confidence")} if word.get("confidence") is not None else {}),
        })
    return output


def _text_from_words(words: Sequence[Mapping[str, Any]]) -> str:
    tokens = [str(word.get("text") or "") for word in words]
    if not tokens:
        return ""
    if any(token[:1].isspace() for token in tokens[1:]):
        return "".join(tokens).strip()
    return " ".join(token.strip() for token in tokens if token.strip()).strip()


def _split_child_id(parent_clip_id: str, side: str, start: float, end: float) -> str:
    digest = hashlib.sha256(
        f"{parent_clip_id}|{side}|{start:.3f}|{end:.3f}".encode()
    ).hexdigest()[:16]
    return f"{parent_clip_id}~{side}-{digest}"


def swap_take(draft: Mapping[str, Any], selected_clip_id: str, replacement_clip_id: str) -> dict[str, Any]:
    out = _copy_draft(draft)
    selected = list(out["selected"])
    alternates = list(out["alternates"])
    selected_lookup = _index(selected)
    alternate_lookup = _index(alternates)
    current = selected_lookup.get(selected_clip_id)
    replacement = alternate_lookup.get(replacement_clip_id)
    if current is None:
        raise DraftEditError("selected clip not found")
    if replacement is None:
        raise DraftEditError("replacement alternate not found")
    current_group = _group_id(current)
    replacement_group = _group_id(replacement)
    if not current_group or current_group != replacement_group:
        raise DraftEditError("replacement must belong to the same take group")

    position = next(index for index, item in enumerate(selected) if _clip_id(item) == selected_clip_id)
    old_item = deepcopy(current)
    old_item["selected"] = False
    new_item = deepcopy(replacement)
    new_item["selected"] = True
    selected[position] = new_item
    alternates = [item for item in alternates if _clip_id(item) != replacement_clip_id]
    alternates.append(old_item)
    out["selected"] = selected
    out["alternates"] = alternates
    return out


def remove_clip(draft: Mapping[str, Any], clip_id: str) -> dict[str, Any]:
    out = _copy_draft(draft)
    selected = list(out["selected"])
    match = next((deepcopy(item) for item in selected if _clip_id(item) == clip_id), None)
    if match is None:
        raise DraftEditError("selected clip not found")
    match["selected"] = False
    out["selected"] = [item for item in selected if _clip_id(item) != clip_id]
    out["alternates"].append(match)
    return out


def restore_clip(draft: Mapping[str, Any], clip_id: str, position: int | None = None) -> dict[str, Any]:
    out = _copy_draft(draft)
    alternates = list(out["alternates"])
    match = next((deepcopy(item) for item in alternates if _clip_id(item) == clip_id), None)
    if match is None:
        raise DraftEditError("alternate clip not found")
    group_id = _group_id(match)
    if group_id and any(_group_id(item) == group_id for item in out["selected"]):
        raise DraftEditError("take group already has a selected clip; use swap instead")
    selected = list(out["selected"])
    insert_at = len(selected) if position is None else min(max(int(position), 0), len(selected))
    match["selected"] = True
    selected.insert(insert_at, match)
    out["selected"] = selected
    out["alternates"] = [item for item in alternates if _clip_id(item) != clip_id]
    return out


def reorder_clips(draft: Mapping[str, Any], ordered_clip_ids: Sequence[str]) -> dict[str, Any]:
    out = _copy_draft(draft)
    selected = list(out["selected"])
    lookup = _index(selected)
    current_ids = [_clip_id(item) for item in selected]
    requested = [str(clip_id) for clip_id in ordered_clip_ids]
    if len(requested) != len(set(requested)) or set(requested) != set(current_ids):
        raise DraftEditError("reorder must contain every selected clip exactly once")
    out["selected"] = [deepcopy(lookup[clip_id]) for clip_id in requested]
    return out


def trim_clip(
    draft: Mapping[str, Any],
    clip_id: str,
    *,
    start: float,
    end: float,
    min_duration_sec: float = MIN_TRIM_DURATION_SEC,
) -> dict[str, Any]:
    out = _copy_draft(draft)
    selected = list(out["selected"])
    position = next((i for i, item in enumerate(selected) if _clip_id(item) == clip_id), None)
    if position is None:
        raise DraftEditError("selected clip not found")
    item = deepcopy(selected[position])
    try:
        current_start = float(item["start"])
        current_end = float(item["end"])
        new_start = float(start)
        new_end = float(end)
    except (KeyError, TypeError, ValueError):
        raise DraftEditError("trim requires numeric clip boundaries") from None
    if current_end <= current_start:
        raise DraftEditError("selected clip has invalid source boundaries")
    if new_start < current_start - 1e-6 or new_end > current_end + 1e-6:
        raise DraftEditError("trim must stay inside the selected clip source interval")
    if new_start < 0 or new_end <= new_start:
        raise DraftEditError("trim end must be after trim start")
    if new_end - new_start < max(0.01, float(min_duration_sec)):
        raise DraftEditError("trim would create an unusable microfragment")
    item["start"] = round(new_start, 3)
    item["end"] = round(new_end, 3)
    selected[position] = item
    out["selected"] = selected
    return out


def split_clip(
    draft: Mapping[str, Any],
    clip_id: str,
    *,
    split_time: float,
    min_duration_sec: float = MIN_SPLIT_DURATION_SEC,
) -> dict[str, Any]:
    out = _copy_draft(draft)
    selected = list(out["selected"])
    position = next((i for i, item in enumerate(selected) if _clip_id(item) == clip_id), None)
    if position is None:
        raise DraftEditError("selected clip not found")
    parent = deepcopy(selected[position])
    try:
        start = float(parent["start"])
        end = float(parent["end"])
        split = float(split_time)
    except (KeyError, TypeError, ValueError):
        raise DraftEditError("split requires numeric clip boundaries") from None
    minimum = max(0.01, float(min_duration_sec))
    if split <= start or split >= end:
        raise DraftEditError("split must be inside the selected clip")
    if split - start < minimum or end - split < minimum:
        raise DraftEditError("split would create an unusable microfragment")

    words = [word for word in _words(parent) if word["end"] >= start and word["start"] <= end]
    transcript = str(parent.get("text") or "")
    caption = str(parent.get("caption_text") if parent.get("caption_text") is not None else transcript)
    if transcript and not words:
        raise DraftEditError("word timings are required to split spoken content safely")
    if transcript and caption != transcript:
        raise DraftEditError("split custom-caption clip before editing its caption text")
    if any(word["start"] < split < word["end"] for word in words):
        raise DraftEditError("split cannot cut through a spoken word")

    left_words = [word for word in words if word["end"] <= split + 1e-6]
    right_words = [word for word in words if word["start"] >= split - 1e-6]
    if transcript and (not left_words or not right_words):
        raise DraftEditError("split must leave spoken words on both sides")
    left_text = _text_from_words(left_words) if words else ""
    right_text = _text_from_words(right_words) if words else ""

    parent_group = parent.get("take_group_id")
    left = deepcopy(parent)
    right = deepcopy(parent)
    left.update({
        "clip_id": _split_child_id(clip_id, "a", start, split),
        "start": round(start, 3),
        "end": round(split, 3),
        "text": left_text,
        "caption_text": left_text,
        "words": left_words,
        "take_group_id": None,
        "parent_clip_id": clip_id,
        "split_from_take_group_id": parent_group,
        "selected": True,
    })
    right.update({
        "clip_id": _split_child_id(clip_id, "b", split, end),
        "start": round(split, 3),
        "end": round(end, 3),
        "text": right_text,
        "caption_text": right_text,
        "words": right_words,
        "take_group_id": None,
        "parent_clip_id": clip_id,
        "split_from_take_group_id": parent_group,
        "selected": True,
    })
    selected[position:position + 1] = [left, right]
    out["selected"] = selected
    return out


def patch_audio(
    draft: Mapping[str, Any],
    clip_id: str,
    *,
    muted: bool | None = None,
    volume: float | None = None,
) -> dict[str, Any]:
    """Update one selected clip's playback/export audio without changing source media."""
    if muted is None and volume is None:
        raise DraftEditError("audio edit requires muted and/or volume")
    out = _copy_draft(draft)
    selected = list(out["selected"])
    position = next((i for i, item in enumerate(selected) if _clip_id(item) == clip_id), None)
    if position is None:
        raise DraftEditError("selected clip not found")
    item = deepcopy(selected[position])
    if muted is not None:
        item["audio_muted"] = bool(muted)
    if volume is not None:
        try:
            resolved = float(volume)
        except (TypeError, ValueError):
            raise DraftEditError("audio volume must be numeric") from None
        if resolved < 0.0 or resolved > 2.0:
            raise DraftEditError("audio volume must be between 0.0 and 2.0")
        item["audio_volume"] = round(resolved, 3)
    selected[position] = item
    out["selected"] = selected
    return out


def patch_captions(draft: Mapping[str, Any], edits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = _copy_draft(draft)
    selected = list(out["selected"])
    lookup = _index(selected)
    edit_ids = [str(edit.get("clip_id") or "") for edit in edits]
    if not edit_ids or any(not clip_id for clip_id in edit_ids):
        raise DraftEditError("caption edits require clip_id")
    if len(edit_ids) != len(set(edit_ids)):
        raise DraftEditError("caption edits must contain unique clip IDs")
    if any(clip_id not in lookup for clip_id in edit_ids):
        raise DraftEditError("caption clip not found in selected draft")
    captions = {str(edit["clip_id"]): str(edit.get("text") or "") for edit in edits}
    for item in selected:
        clip_id = _clip_id(item)
        if clip_id in captions:
            item["caption_text"] = captions[clip_id]
    out["selected"] = selected
    return out
