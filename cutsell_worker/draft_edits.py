"""Pure stateless edits for the CutSell mobile Draft Timeline.

Every operation returns a new JSON-safe draft and never mutates its input. The
mobile client can therefore keep its own undo/redo stack while persistence is built
later. Commercial meaning never grants deletion authority here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


class DraftEditError(ValueError):
    pass


MIN_TRIM_DURATION_SEC = 0.15


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
    """Trim one selected clip inward without changing source identity or transcript."""
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
    # Transcript/caption strings are not rewritten from guessed text. The media
    # boundaries change; a later word-aware caption timing layer can narrow timing.
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
            # Transcript text remains immutable; only the user-facing caption changes.
            item["caption_text"] = captions[clip_id]
    out["selected"] = selected
    return out
