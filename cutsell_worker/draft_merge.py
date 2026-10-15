"""Editor v5: reprocessing a project must not wipe the creator's edits.

`merge_reprocessed_draft(old, new)` is the ONE pure function that combines the project's stored
draft (`old`, possibly edited in the iPhone editor) with the draft a new cut just produced (`new`):

- when the creator has already edited the stored draft (`creator_edited`), their cut is NEVER
  replaced: only the AI zoom keyframes and `diagnostics` are refreshed (`keep_creator_cut`) --
  a re-run of the same job after a worker restart must not undo deletions, splits or reorders;
- on an untouched draft, the engine's new suggestions win for the cut itself: `selected`, `alternates`, `discarded`,
  `diagnostics` (and every other engine field);
- the creator's editor settings are kept exactly as they were (`EDITOR_FIELDS` and every
  `caption_*` key) -- a key the old draft did not have stays at its default;
- each new clip takes the per-clip edits (`CLIP_EDIT_FIELDS`, and a caption edited by hand) of the
  old clip from the same source whose window overlaps it most, when that overlap covers at least
  half of the new clip. No clip is ever invented or removed;
- B-roll clips the creator placed (source listed in the old `broll_sources`) are put back,
  unchanged, at their old timeline time: at a clip boundary, by splitting the clip they land in
  (as the app's insertBroll does), or appended past the end;
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


BOUNDARY_TOLERANCE_SEC = 0.05


def _speed(clip: Mapping[str, Any]) -> float:
    try:
        value = float(clip.get("speed", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return value if value > 0 else 1.0


def _timeline_length(clip: Mapping[str, Any]) -> float:
    window = _window(clip)
    return 0.0 if window is None else (window[1] - window[0]) / _speed(clip)


def _text_from_words(words: list) -> str:
    return " ".join(str(w.get("text") or "").strip() for w in words if str(w.get("text") or "").strip())


def _split_for_broll(clip: dict[str, Any], source_split: float, broll_id: str) -> tuple[dict, dict]:
    """Split one clip at a source time, the way the app's insertBroll does: words go to the side
    their middle falls on, text/caption re-derived (a caption edited by hand is kept as is), the
    right piece is `<id>-<last4 of the B-roll clip id>`, the left piece loses `manual_end`."""
    hand_caption = _caption_edited(clip)
    left, right = deepcopy(clip), deepcopy(clip)
    left["end"] = round(source_split, 3)
    right["start"] = round(source_split, 3)
    words = [w for w in clip.get("words") or () if isinstance(w, dict)]
    if "words" in clip:
        left_words, right_words = [], []
        for word in words:
            try:
                middle = (float(word["start"]) + float(word["end"])) / 2.0
            except (KeyError, TypeError, ValueError):
                continue
            (left_words if middle < source_split else right_words).append(deepcopy(word))
        left["words"], right["words"] = left_words, right_words
        if words:
            left["text"], right["text"] = _text_from_words(left_words), _text_from_words(right_words)
            if not hand_caption:
                left["caption_text"], right["caption_text"] = left["text"], right["text"]
    right["clip_id"] = f"{clip.get('clip_id')}-{str(broll_id)[-4:]}"
    left.pop("manual_end", None)
    return left, right


def _reinsert_broll(selected: list[dict[str, Any]], old: Mapping[str, Any], report: list) -> list[dict[str, Any]]:
    """Put every B-roll clip the creator placed (its source is in the old `broll_sources`) back at
    its old timeline time, unchanged: at a clip boundary it goes there, inside a clip that clip is
    split, past the end it is appended."""
    broll_ids = {str(item.get("source_asset_id")) for item in old.get("broll_sources") or ()
                 if isinstance(item, dict) and item.get("source_asset_id")}
    if not broll_ids:
        return selected
    placed = []
    position = 0.0
    for clip in old.get("selected") or ():
        if not isinstance(clip, dict):
            continue
        if str(clip.get("source_asset_id")) in broll_ids:
            placed.append((position, clip))
        position += _timeline_length(clip)
    out = list(selected)
    present = {str(c.get("clip_id")) for c in out}
    for time, broll in placed:  # in old timeline order, so earlier re-inserts are counted
        if str(broll.get("clip_id")) in present:
            continue
        cursor, index, action = 0.0, len(out), "appended_after_end"
        for i, clip in enumerate(out):
            length = _timeline_length(clip)
            if abs(time - cursor) <= BOUNDARY_TOLERANCE_SEC:
                index, action = i, "inserted_at_boundary"
                break
            if cursor < time < cursor + length - BOUNDARY_TOLERANCE_SEC:
                window = _window(clip)
                source_split = window[0] + (time - cursor) * _speed(clip)
                left, right = _split_for_broll(clip, source_split, str(broll.get("clip_id") or ""))
                out[i:i + 1] = [left, right]
                index, action = i + 1, "split_clip"
                break
            cursor += length
        else:
            if abs(time - cursor) <= BOUNDARY_TOLERANCE_SEC:
                action = "inserted_at_boundary"
        out.insert(index, deepcopy(broll))
        present.add(str(broll.get("clip_id")))
        report.append({"clip_id": broll.get("clip_id"), "timeline_sec": round(time, 3), "action": action})
    return out


def _merge_zooms(old: Mapping[str, Any], new: Mapping[str, Any]) -> list:
    tombstones = list(old.get("zoom_removed") or ())
    user = [deepcopy(k) for k in old.get("zoom_keyframes") or () if isinstance(k, dict)
            and str(k.get("origin") or "ai") != "ai"]
    ai = [deepcopy(k) for k in new.get("zoom_keyframes") or () if isinstance(k, dict)
          and str(k.get("origin") or "ai") == "ai" and not is_tombstoned_zoom(k, tombstones)]
    return user + ai


def keep_creator_cut(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """The creator has edited this draft: their cut (`selected`, `alternates`, `discarded`) and
    every editor setting stay exactly as they are; only what belongs to the AI is refreshed --
    the AI zoom keyframes (user ones kept, deleted ones never back) and `diagnostics`."""
    merged: dict[str, Any] = deepcopy(dict(old))
    zooms = _merge_zooms(old, new)
    if zooms or "zoom_keyframes" in merged:
        merged["zoom_keyframes"] = zooms
    diagnostics = deepcopy(dict(new.get("diagnostics") or {}))
    diagnostics["reprocess_merge"] = {"merged_with_previous_draft": True, "kept_creator_cut": True}
    merged["diagnostics"] = diagnostics
    return merged


def _without_merge_note(draft: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(draft)
    diagnostics = dict(out.get("diagnostics") or {})
    diagnostics.pop("reprocess_merge", None)
    out["diagnostics"] = diagnostics
    return out


def changes_anything(old: Mapping[str, Any], merged: Mapping[str, Any]) -> bool:
    """False when the merge only re-adds its own bookkeeping note (nothing to save)."""
    return _without_merge_note(old) != _without_merge_note(merged)


def merge_reprocessed_draft(old: Mapping[str, Any] | None, new: Mapping[str, Any], *,
                            creator_edited: bool = False) -> dict[str, Any]:
    """See the module docstring. `old` None/empty (first processing) returns `new` unchanged.
    `creator_edited` (the stored draft has revisions/undo from the editor -- e.g. the same job
    running again after a worker restart): the creator's cut is kept untouched (`keep_creator_cut`)."""
    if not old or not isinstance(old, Mapping):
        return deepcopy(dict(new))
    if creator_edited:
        return keep_creator_cut(old, new)
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

    # 2b. B-roll the creator placed by hand is never an engine suggestion: put it back
    broll_rows: list = []
    if "selected" in merged:
        merged["selected"] = _reinsert_broll(list(merged["selected"]), old, broll_rows)

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
        "broll_reinserted": broll_rows,
        "timed_item_adjustments": adjustments,
    }
    merged["diagnostics"] = diagnostics
    return merged
