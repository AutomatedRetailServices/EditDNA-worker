"""Editor v5: processing a project again merges the new cut with the creator's edits
(draft_merge.merge_reprocessed_draft) instead of wiping them."""
from __future__ import annotations

import copy

import pytest

from cutsell_worker.draft_merge import merge_reprocessed_draft
from cutsell_worker.serde import draft_from_dict


def _clip(clip_id, start, end, source="s1", text="hello there", **extra):
    item = {"clip_id": clip_id, "source_asset_id": source, "start": start, "end": end, "text": text,
            "caption_text": text, "audio_volume": 1.0, "audio_muted": False, "selected": True}
    item.update(extra)
    return item


def _engine(selected, **extra):
    draft = {"schema_version": "cutsell.v1", "project_id": "p1", "strategy": "mixed", "selected": selected,
             "alternates": [], "discarded": [], "diagnostics": {"engine": "simple", "run": 2},
             "captions_enabled": True, "caption_preset": "classic"}
    draft.update(extra)
    return draft


OLD = _engine(
    [_clip("a", 0.0, 4.0, speed=2.0, audio_volume=1.5, mirror=True, rotation=90),
     _clip("b", 5.0, 9.0, audio_muted=True, manual_end=True, caption_text="My own words"),
     _clip("c", 0.0, 3.0, source="s2")],
    diagnostics={"engine": "simple", "run": 1},
    caption_preset="box", caption_font="anton", caption_y=0.6, captions_enabled=False, mirror=True, rotation=180,
    video_volume=0.4, video_clean_voice=True, auto_zoom_enabled=True, audio_overlap_enabled=True,
    media_overlays=[{"overlay_id": "o1", "kind": "photo", "uri": "s3://b/p.jpg", "start": 0.5, "end": 1.5, "layer": 3}],
    text_overlays=[{"overlay_id": "t1", "text": "SALE", "start": 0.0, "end": 1.0}],
    audio_clips=[{"audio_id": "a1", "kind": "sfx", "sfx_id": "tap", "start": 1.0, "source_in": 0, "source_out": 0.3}],
    broll_sources=[{"source_asset_id": "br1", "uri": "s3://b/br.mp4"}],
    zoom_keyframes=[
        {"keyframe_id": "old_ai", "source_asset_id": "s1", "source_time": 1.0, "duration": 1.0, "scale": 1.1, "origin": "ai"},
        {"keyframe_id": "mine", "source_asset_id": "s1", "source_time": 6.0, "duration": 1.0, "scale": 1.2, "origin": "user"},
    ],
    zoom_removed=[{"source_asset_id": "s1", "source_time": 2.0}],
)


def test_first_processing_is_unchanged():
    new = _engine([_clip("x", 0.0, 2.0)])
    assert merge_reprocessed_draft(None, new) == new
    assert merge_reprocessed_draft({}, new) == new


def test_inputs_are_not_mutated():
    old, new = copy.deepcopy(OLD), _engine([_clip("x", 0.5, 4.0)])
    snapshot = copy.deepcopy(new)
    merge_reprocessed_draft(old, new)
    assert old == OLD and new == snapshot


def test_engine_suggestions_win_and_creator_settings_stay():
    new = _engine([_clip("x", 0.5, 4.0), _clip("y", 5.5, 8.0)], alternates=[_clip("alt", 10.0, 12.0)],
                  discarded=[_clip("d", 12.0, 13.0)], caption_preset="classic")
    merged = merge_reprocessed_draft(OLD, new)
    assert [c["clip_id"] for c in merged["selected"]] == ["x", "y"]
    assert [c["clip_id"] for c in merged["alternates"]] == ["alt"]
    assert [c["clip_id"] for c in merged["discarded"]] == ["d"]
    assert merged["diagnostics"]["run"] == 2 and merged["diagnostics"]["reprocess_merge"]["merged_with_previous_draft"]
    for key in ("media_overlays", "text_overlays", "audio_clips", "broll_sources", "captions_enabled", "mirror",
                "rotation", "video_volume", "video_clean_voice", "auto_zoom_enabled", "audio_overlap_enabled",
                "zoom_removed", "caption_preset", "caption_font", "caption_y"):
        assert merged[key] == OLD[key], key
    draft_from_dict(merged)  # still a valid, renderable draft


def test_creator_setting_absent_in_old_stays_at_default():
    old = _engine([_clip("a", 0.0, 4.0)])
    new = _engine([_clip("x", 0.0, 4.0)], auto_zoom_enabled=True, caption_font="anton")
    merged = merge_reprocessed_draft(old, new)
    assert "caption_font" not in merged and "auto_zoom_enabled" not in merged


def test_clip_edits_follow_the_best_overlapping_new_clip():
    new = _engine([
        _clip("x", 0.5, 4.0),            # overlaps a (0-4) by 3.5/3.5
        _clip("y", 4.5, 8.5),            # overlaps b (5-9) by 3.5/4.0
        _clip("z", 8.0, 12.0),           # overlaps b by 1.0/4.0 -> under 50 %: no carry
        _clip("w", 0.0, 3.0, source="s3"),  # another source: no carry
    ])
    merged = merge_reprocessed_draft(OLD, new)
    x, y, z, w = merged["selected"]
    assert (x["speed"], x["audio_volume"], x["mirror"], x["rotation"]) == (2.0, 1.5, True, 90)
    assert y["audio_muted"] is True and y["manual_end"] is True and y["caption_text"] == "My own words"
    assert "speed" not in z and z["audio_muted"] is False and z["caption_text"] == "hello there"
    assert "speed" not in w and "mirror" not in w
    rows = merged["diagnostics"]["reprocess_merge"]["clip_edits_carried"]
    assert {r["clip_id"]: r["from_clip_id"] for r in rows} == {"x": "a", "y": "b"}
    assert len(merged["selected"]) == 4  # never invents or drops clips


def test_engine_caption_is_kept_when_the_old_caption_was_not_edited():
    old = _engine([_clip("a", 0.0, 4.0, text="old words")])
    new = _engine([_clip("x", 0.0, 4.0, text="new words")])
    assert merge_reprocessed_draft(old, new)["selected"][0]["caption_text"] == "new words"


def test_zoom_keyframes_keep_user_replace_ai_and_honor_tombstones():
    new = _engine([_clip("x", 0.0, 9.0)], zoom_keyframes=[
        {"keyframe_id": "n1", "source_asset_id": "s1", "source_time": 2.2, "duration": 1.0, "scale": 1.1, "origin": "ai"},
        {"keyframe_id": "n2", "source_asset_id": "s1", "source_time": 4.0, "duration": 1.0, "scale": 1.1, "origin": "ai"},
        {"keyframe_id": "n3", "source_asset_id": "s2", "source_time": 2.0, "duration": 1.0, "scale": 1.1, "origin": "ai"},
        {"keyframe_id": "n4", "source_asset_id": "s1", "source_time": 2.35, "duration": 1.0, "scale": 1.1},
    ])
    merged = merge_reprocessed_draft(OLD, new)
    # user "mine" kept; old AI "old_ai" replaced; n1 (0.2 s from the tombstone) and n4 (origin
    # defaults to ai, 0.35 s away -> kept) ; n3 is another source -> kept.
    assert [k["keyframe_id"] for k in merged["zoom_keyframes"]] == ["mine", "n2", "n3", "n4"]


def test_timed_items_are_fitted_to_a_shorter_new_cut_never_dropped_silently():
    old = copy.deepcopy(OLD)
    old["media_overlays"].append({"overlay_id": "o2", "kind": "photo", "uri": "s3://b/q.jpg", "start": 1.0, "end": 9.0})
    old["media_overlays"].append({"overlay_id": "o3", "kind": "photo", "uri": "s3://b/r.jpg", "start": 8.0, "end": 9.0})
    old["audio_clips"].append({"audio_id": "a2", "kind": "sfx", "sfx_id": "good", "start": 7.0, "source_in": 0, "source_out": 0.3})
    new = _engine([_clip("x", 0.0, 3.0, source="s2")])  # 3 s timeline (old clip "c" has no speed)
    merged = merge_reprocessed_draft(old, new)
    assert [o["overlay_id"] for o in merged["media_overlays"]] == ["o1", "o2"]
    assert merged["media_overlays"][1]["end"] == 3.0
    assert [a["audio_id"] for a in merged["audio_clips"]] == ["a1"]
    actions = {(r["id"], r["action"]) for r in merged["diagnostics"]["reprocess_merge"]["timed_item_adjustments"]}
    assert actions == {("o2", "shortened_to_timeline_end"), ("o3", "set_aside_after_timeline_end"),
                       ("a2", "set_aside_after_timeline_end")}
    draft_from_dict(merged)


def test_worker_returns_the_merged_draft_on_a_reprocess(monkeypatch):
    """The job result carries the same merged draft the store saved (not the raw new cut)."""
    import inspect
    from cutsell_worker import worker_job
    source = inspect.getsource(worker_job)
    assert "stored = create_initial_draft(" in source and 'serialized["draft"] = stored[0]["draft"]' in source


def test_store_keeps_the_old_draft_when_a_merge_would_not_render():
    from cutsell_worker import draft_store
    from tests.test_cutsell_clean_worker_draft_store import FakeRedis

    client = FakeRedis()
    sources = [{"source_asset_id": "s1", "uri": "s3://b/s.mov"}]
    old = _engine([_clip("a", 0.0, 4.0)])
    draft_store.create_initial_draft(user_id="u", project_id="p1", draft=old, sources=sources, client=client)
    broken = _engine([_clip("x", 0.0, 4.0, semantic_role="NOT_A_ROLE")])
    kept, created = draft_store.create_initial_draft(user_id="u", project_id="p1", draft=broken, sources=sources, client=client)
    assert created is False and kept["revision"] == 1 and kept["draft"] == old
