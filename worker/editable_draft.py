"""Stable editable Flow B draft contract shared by web and mobile clients."""
from typing import Any, Dict, Iterable, List, Mapping, Sequence

DRAFT_SCHEMA_VERSION = "v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _public_clip(clip: Mapping[str, Any], *, selected: bool) -> Dict[str, Any]:
    meta = clip.get("meta") if isinstance(clip.get("meta"), Mapping) else {}
    semantic_v2 = meta.get("semantic_v2") if isinstance(meta.get("semantic_v2"), Mapping) else {}
    return {
        "clip_id": str(clip.get("id") or ""),
        "source_index": int(clip.get("source_index", 0)),
        "start": _safe_float(clip.get("source_start", clip.get("start", 0.0))),
        "end": _safe_float(clip.get("source_end", clip.get("end", clip.get("start", 0.0)))),
        "text": str(clip.get("text") or ""),
        "slot": str(clip.get("slot") or "OTHER"),
        "selected": selected,
        "clean_cut_keep": bool(meta.get("keep", True)),
        "semantic_score": _safe_float(clip.get("semantic_score", 0.0)),
        "visual_score": _safe_float(clip.get("visual_score", 0.0)),
        "score": _safe_float(clip.get("score", clip.get("semantic_score", 0.0))),
        "semantic_v2": {
            "applied": bool(semantic_v2.get("applied", False)),
            "primary_slot": semantic_v2.get("primary_slot"),
            "confidence": semantic_v2.get("confidence"),
            "abstain": semantic_v2.get("abstain"),
        },
        "take_judge_status": meta.get("take_judge_execution_status"),
        "take_judge_selected": bool(meta.get("take_judge_selected", False)),
    }


def build_editable_draft(
    clips: Sequence[Mapping[str, Any]],
    selected_clip_ids: Iterable[str],
    *,
    mode: str,
    clean_cut_discard_diagnostics: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Create a fail-open draft without mutating pipeline clip state.

    Clean Cut decides whether a candidate is valid. The composer/Best Take decides
    whether a valid candidate is selected. Valid but unselected candidates remain
    available as alternates for Swap/Restore in the product UI.
    """
    selected_order = list(dict.fromkeys(str(value) for value in selected_clip_ids if value))
    selected_set = set(selected_order)
    by_id = {str(clip.get("id") or ""): clip for clip in clips if clip.get("id")}

    selected: List[Dict[str, Any]] = []
    for clip_id in selected_order:
        clip = by_id.get(clip_id)
        if clip is None:
            continue
        meta = clip.get("meta") if isinstance(clip.get("meta"), Mapping) else {}
        if not meta.get("keep", True):
            continue
        selected.append(_public_clip(clip, selected=True))

    alternates: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    for clip in clips:
        clip_id = str(clip.get("id") or "")
        if not clip_id:
            continue
        meta = clip.get("meta") if isinstance(clip.get("meta"), Mapping) else {}
        if not meta.get("keep", True):
            discarded.append(_public_clip(clip, selected=False))
        elif clip_id not in selected_set:
            alternates.append(_public_clip(clip, selected=False))

    boundary_discards = []
    for item in clean_cut_discard_diagnostics or ():
        boundary_discards.append({
            "diagnostic_id": str(item.get("diagnostic_id") or ""),
            "clip_id": str(item.get("clip_id") or ""),
            "source_index": int(item.get("source_index", 0)),
            "start": _safe_float(item.get("source_start", item.get("start", 0.0))),
            "end": _safe_float(item.get("source_end", item.get("end", item.get("start", 0.0)))),
            "text": str(item.get("text") or ""),
            "reason": str(item.get("reason") or "clean_cut_discard"),
        })

    return {
        "schema_version": DRAFT_SCHEMA_VERSION,
        "mode": str(mode or "human"),
        "selected_clip_ids": [item["clip_id"] for item in selected],
        "selected": selected,
        "alternates": alternates,
        "discarded": discarded,
        "boundary_discards": boundary_discards,
        "counts": {
            "selected": len(selected),
            "alternates": len(alternates),
            "discarded": len(discarded) + len(boundary_discards),
        },
    }
