"""Editor v5: reprocessing a project must not wipe the creator's edits.

`merge_reprocessed_draft(old, new)` is the ONE pure function that combines the project's stored
draft (`old`, possibly edited in the iPhone editor) with the draft a new cut just produced (`new`):

- the engine's new suggestions win for the cut itself: `selected`, `alternates`, `discarded`,
  `diagnostics` (and every other engine field);
- the creator's editor settings are kept exactly as they were (`EDITOR_FIELDS` and every
  `caption_*` key) -- a key the old draft did not have stays at its default;
- each new clip takes the per-clip edits (`CLIP_EDIT_FIELDS`, and a caption edited by hand) of the
  old clip from the same source whose window overlaps it most, when that overlap covers at least
  half of the new clip. No clip is ever invented or removed;
- zoom keyframes: every creator keyframe (origin other than "ai") is kept; old AI keyframes are
  replaced by the new cut's AI keyframes, minus any within 0.3 s (same source) of a `zoom_removed`
  tombstone;
- overlays / audio clips that would now run past the (possibly shorter) new timeline are
  shortened to it, or set aside in `diagnostics["reprocess_merge"]` when they would start after
  its end, so the merged draft always stays valid. Nothing is dropped silently.

Inputs are never mutated.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .render_plan import is_tombstoned_zoom

#: Draft-level settings the creator owns. A reprocess keeps them as they were.
EDITOR_FIELDS: tuple[str, ...] = (
    "media_overlays", "text_overlays", "audio_clips", "broll_sources", "captions_enabled", "mirror",
    "rotation", "video_volume", "video_clean_voice", "auto_zoom_enabled", "audio_overlap_enabled",
    "zoom_removed",
)
#: Per-clip edits carried to the matching new clip.
CLIP_EDIT_FIELDS: tuple[str, ...] = ("speed", "audio_volume", "audio_muted", "mirror", "rotation", "manual_end")
MIN_OVERLAP_SHARE = 0.5
_EPS = 1e-6


def _is_editor_key(key: str) -> bool:
    return key in EDITOR_FIELDS or key.startswith("caption_")


def _window(clip: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        start, end = float(clip["start"]), float(clip["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _caption_edited(clip: Mapping[str, Any]) -> bool:
    caption = clip.get("caption_text")
    if caption is None:
        return False
    return " ".join(str(caption).split()) != " ".join(str(clip.get("text") or "").split())


def _best_old_clip(new_clip: Mapping[str, Any], old_clips: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    window = _window(new_clip)
    if window is None:
        return None
    best, best_overlap = None, 0.0
    for old in old_clips:
        if str(old.get("source_asset_id")) != str(new_clip.get("source_asset_id")):
            continue
        old_window = _window(old)
        if old_window is None:
            continue
        overlap = min(window[1], old_window[1]) - max(window[0], old_window[0])
        if overlap > best_overlap + _EPS:
            best, best_overlap = old, overlap
    if best is None or best_overlap + _EPS < MIN_OVERLAP_SHARE * (window[1] - window[0]):
        return None
    return best


def _carry_clip_edits(new_clip: dict[str, Any], old_clip: Mapping[str, Any]) -> list[str]:
    carried = []
    for key in CLIP_EDIT_FIELDS:
        if key in old_clip:
            if new_clip.get(key) != old_clip[key]:
                carried.append(key)
            new_clip[key] = deepcopy(old_clip[key])
    if _caption_edited(old_clip):
        if new_clip.get("caption_text") != old_clip.get("caption_text"):
            carried.append("caption_text")
        new_clip["caption_text"] = old_clip.get("caption_text")
    return carried


def _timeline_duration(selected) -> float:
    total = 0.0
    for clip in selected or ():
        window = _window(clip)
        if window is None:
            continue
        try:
            speed = float(clip.get("speed", 1.0) or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        total += (window[1] - window[0]) / (speed if speed > 0 else 1.0)
    return total


def _fit_timed_items(items, duration: float, *, kind: str, end_key: str | None, report: list) -> list:
    """Keep every item that still starts inside the new timeline (shortening an overlay that would
    end after it); set aside, with a report row, any that would start after it."""
    kept = []
    for item in items or ():
        if not isinstance(item, dict):
            continue
        entry = deepcopy(item)
        ident = entry.get("overlay_id") or entry.get("audio_id")
        try:
            start = float(entry.get("start", 0.0))
        except (TypeError, ValueError):
            kept.append(entry)
            continue
        if kind == "audio_clips":
            if start > duration + 0.05:
                report.append({"kind": kind, "id": ident, "action": "set_aside_after_timeline_end", "item": entry})
                continue
            kept.append(entry)
            continue
        if start >= duration - 0.05:
            report.append({"kind": kind, "id": ident, "action": "set_aside_after_timeline_end", "item": entry})
            continue
        try:
            end = float(entry.get(end_key))
        except (TypeError, ValueError):
            kept.append(entry)
            continue
        if end > duration + _EPS:
            entry[end_key] = round(duration, 3)
            report.append({"kind": kind, "id": ident, "action": "shortened_to_timeline_end",
                           "original_end": end, "end": entry[end_key]})
        kept.append(entry)
    return kept


def _merge_zooms(old: Mapping[str, Any], new: Mapping[str, Any]) -> list:
    tombstones = list(old.get("zoom_removed") or ())
    user = [deepcopy(k) for k in old.get("zoom_keyframes") or () if isinstance(k, dict)
            and str(k.get("origin") or "ai") != "ai"]
    ai = [deepcopy(k) for k in new.get("zoom_keyframes") or () if isinstance(k, dict)
          and str(k.get("origin") or "ai") == "ai" and not is_tombstoned_zoom(k, tombstones)]
    return user + ai


def merge_reprocessed_draft(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> dict[str, Any]:
    """See the module docstring. `old` None/empty (first processing) returns `new` unchanged."""
    if not old or not isinstance(old, Mapping):
        return deepcopy(dict(new))
    merged: dict[str, Any] = deepcopy(dict(new))
    # 1. creator-owned draft settings, exactly as they were
    for key in set(merged) | set(old):
        if not _is_editor_key(key):
            continue
        if key in old:
            merged[key] = deepcopy(old[key])
        else:
            merged.pop(key, None)

    # 2. per-clip edits onto the matching new clips
    old_clips = [c for c in old.get("selected") or () if isinstance(c, dict)]
    carried_rows = []
    for list_key in ("selected", "alternates", "discarded"):
        clips = []
        for clip in merged.get(list_key) or ():
            clip = dict(clip)
            match = _best_old_clip(clip, old_clips)
            if match is not None:
                fields = _carry_clip_edits(clip, match)
                if fields and list_key == "selected":
                    carried_rows.append({"clip_id": clip.get("clip_id"), "from_clip_id": match.get("clip_id"),
                                         "fields": fields})
            clips.append(clip)
        if list_key in merged:
            merged[list_key] = clips

    # 3. zoom keyframes
    zooms = _merge_zooms(old, new)
    if zooms or "zoom_keyframes" in merged:
        merged["zoom_keyframes"] = zooms

    # 4. keep the creator's timed items valid on the new timeline
    duration = _timeline_duration(merged.get("selected"))
    adjustments: list = []
    for key in ("media_overlays", "text_overlays"):
        if key in merged:
            merged[key] = _fit_timed_items(merged[key], duration, kind=key, end_key="end", report=adjustments)
    if "audio_clips" in merged:
        merged["audio_clips"] = _fit_timed_items(merged["audio_clips"], duration, kind="audio_clips",
                                                 end_key=None, report=adjustments)

    diagnostics = dict(merged.get("diagnostics") or {})
    diagnostics["reprocess_merge"] = {
        "merged_with_previous_draft": True,
        "clip_edits_carried": carried_rows,
        "timed_item_adjustments": adjustments,
    }
    merged["diagnostics"] = diagnostics
    return merged
