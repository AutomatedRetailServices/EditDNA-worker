"""Pure editable-draft operations for Flow A/Flow B V1.

Persistence and authorization live outside this module. Every operation returns a
new draft and never mutates its input, making retries/idempotency safe at the API layer.
"""
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Sequence


class DraftEditError(ValueError):
    pass


def _clip_id(item: Mapping[str, Any]) -> str:
    return str(item.get("clip_id") or "")


def _index(items: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {_clip_id(item): item for item in items if _clip_id(item)}


def _recount(draft: Dict[str, Any]) -> None:
    draft["selected_clip_ids"] = [_clip_id(item) for item in draft.get("selected", [])]
    draft["counts"] = {
        "selected": len(draft.get("selected", [])),
        "alternates": len(draft.get("alternates", [])),
        "discarded": len(draft.get("discarded", [])) + len(draft.get("boundary_discards", [])),
    }


def swap_take(draft: Mapping[str, Any], selected_clip_id: str, replacement_clip_id: str) -> Dict[str, Any]:
    out = deepcopy(dict(draft))
    selected = list(out.get("selected", []))
    alternates = list(out.get("alternates", []))
    selected_lookup = _index(selected)
    alternate_lookup = _index(alternates)
    if selected_clip_id not in selected_lookup:
        raise DraftEditError("selected clip not found")
    if replacement_clip_id not in alternate_lookup:
        raise DraftEditError("replacement alternate not found")

    position = next(i for i, item in enumerate(selected) if _clip_id(item) == selected_clip_id)
    old_item = deepcopy(selected_lookup[selected_clip_id]); old_item["selected"] = False
    new_item = deepcopy(alternate_lookup[replacement_clip_id]); new_item["selected"] = True
    selected[position] = new_item
    alternates = [item for item in alternates if _clip_id(item) != replacement_clip_id]
    alternates.append(old_item)
    out["selected"], out["alternates"] = selected, alternates
    _recount(out)
    return out


def remove_clip(draft: Mapping[str, Any], clip_id: str) -> Dict[str, Any]:
    out = deepcopy(dict(draft))
    selected = list(out.get("selected", []))
    match = next((deepcopy(item) for item in selected if _clip_id(item) == clip_id), None)
    if match is None:
        raise DraftEditError("selected clip not found")
    match["selected"] = False
    out["selected"] = [item for item in selected if _clip_id(item) != clip_id]
    out.setdefault("alternates", []).append(match)
    _recount(out)
    return out


def restore_clip(draft: Mapping[str, Any], clip_id: str, position: Optional[int] = None) -> Dict[str, Any]:
    out = deepcopy(dict(draft))
    alternates = list(out.get("alternates", []))
    match = next((deepcopy(item) for item in alternates if _clip_id(item) == clip_id), None)
    if match is None:
        raise DraftEditError("alternate clip not found")
    match["selected"] = True
    selected = list(out.get("selected", []))
    index = len(selected) if position is None else min(max(position, 0), len(selected))
    selected.insert(index, match)
    out["selected"] = selected
    out["alternates"] = [item for item in alternates if _clip_id(item) != clip_id]
    _recount(out)
    return out


def reorder_clips(draft: Mapping[str, Any], ordered_clip_ids: Sequence[str]) -> Dict[str, Any]:
    out = deepcopy(dict(draft))
    selected = list(out.get("selected", []))
    lookup = _index(selected)
    current_ids = [_clip_id(item) for item in selected]
    requested = list(ordered_clip_ids)
    if len(requested) != len(set(requested)) or set(requested) != set(current_ids):
        raise DraftEditError("reorder must contain every selected clip exactly once")
    out["selected"] = [deepcopy(lookup[clip_id]) for clip_id in requested]
    _recount(out)
    return out
