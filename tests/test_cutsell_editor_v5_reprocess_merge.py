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


# --- B-roll the creator placed is kept --------------------------------------------------------------

def _broll(clip_id="broll_abcd1234", start=0.0, end=2.0, **extra):
    item = {"clip_id": clip_id, "source_asset_id": "src_broll_1", "source_order": 1, "start": start, "end": end,
            "text": "", "caption_text": "", "words": [], "audio_volume": 0.5, "audio_muted": False, "selected": True}
    item.update(extra)
    return item


def _words(*spans):
    return [{"text": text, "start": a, "end": b} for text, a, b in spans]


def _timeline_starts(selected):
    out, cursor = {}, 0.0
    for clip in selected:
        out[clip["clip_id"]] = round(cursor, 3)
        cursor += (clip["end"] - clip["start"]) / float(clip.get("speed", 1.0) or 1.0)
    return out


BROLL_SOURCES = [{"source_asset_id": "src_broll_1", "uri": "s3://b/cutsell/uploads/x/y/br.mp4"}]


def test_broll_in_the_middle_of_a_new_clip_splits_it():
    broll = _broll(speed=2.0)  # 1 s on the timeline
    old = _engine([_clip("a", 0.0, 3.0), broll, _clip("b", 4.0, 6.0)], broll_sources=BROLL_SOURCES)
    # The new cut has one long clip where the B-roll used to start (timeline 3.0 s).
    new_clip = _clip("x", 0.0, 6.0, text="one two three four", caption_text="one two three four",
                     words=_words(("one", 0.5, 1.0), ("two", 2.0, 2.5), ("three", 3.2, 3.8), ("four", 4.5, 5.0)),
                     manual_end=True)
    merged = merge_reprocessed_draft(old, _engine([new_clip]))
    ids = [c["clip_id"] for c in merged["selected"]]
    assert ids == ["x", "broll_abcd1234", "x-1234"]
    left, placed, right = merged["selected"]
    assert placed == broll                                              # unchanged
    assert _timeline_starts(merged["selected"])["broll_abcd1234"] == 3.0  # same timeline time
    assert (left["start"], left["end"], right["start"], right["end"]) == (0.0, 3.0, 3.0, 6.0)
    assert left["text"] == left["caption_text"] == "one two" and right["text"] == right["caption_text"] == "three four"
    assert [w["text"] for w in right["words"]] == ["three", "four"]
    assert "manual_end" not in left and right["manual_end"] is True
    assert merged["broll_sources"] == BROLL_SOURCES
    parsed = draft_from_dict(merged)
    assert parsed.user_media_source_ids == ("src_broll_1",)
    assert merged["diagnostics"]["reprocess_merge"]["broll_reinserted"] == [
        {"clip_id": "broll_abcd1234", "timeline_sec": 3.0, "action": "split_clip"}]


def test_broll_split_keeps_a_hand_edited_caption():
    old = _engine([_clip("a", 0.0, 2.0), _broll(), _clip("b", 2.0, 4.0)], broll_sources=BROLL_SOURCES)
    new_clip = _clip("x", 0.0, 4.0, text="one two", caption_text="My caption",
                     words=_words(("one", 0.5, 1.0), ("two", 3.0, 3.5)))
    merged = merge_reprocessed_draft(old, _engine([new_clip]))
    left, _placed, right = merged["selected"]
    assert left["text"] == "one" and right["text"] == "two"
    assert left["caption_text"] == right["caption_text"] == "My caption"
    draft_from_dict(merged)


def test_broll_at_a_clip_boundary_and_past_the_end():
    first, last = _broll("broll_first_aaaa", 0.0, 1.0), _broll("broll_last_bbbb", 1.0, 2.5)
    old = _engine([_clip("a", 0.0, 2.0), first, _clip("b", 3.0, 5.0), last], broll_sources=BROLL_SOURCES)
    # old timeline: a 0-2, first 2-3, b 3-5, last 5-6.5. New cut: x (2 s) and y (1 s): 3 s long.
    merged = merge_reprocessed_draft(old, _engine([_clip("x", 0.0, 2.0), _clip("y", 6.0, 7.0)]))
    assert [c["clip_id"] for c in merged["selected"]] == ["x", "broll_first_aaaa", "y", "broll_last_bbbb"]
    starts = _timeline_starts(merged["selected"])
    assert starts["broll_first_aaaa"] == 2.0          # boundary: same time
    assert merged["selected"][-1] == last              # past the end: appended, unchanged
    actions = [r["action"] for r in merged["diagnostics"]["reprocess_merge"]["broll_reinserted"]]
    assert actions == ["inserted_at_boundary", "appended_after_end"]
    draft_from_dict(merged)


def test_broll_at_the_exact_end_is_appended_and_no_broll_means_no_change():
    old = _engine([_clip("a", 0.0, 2.0), _broll()], broll_sources=BROLL_SOURCES)
    merged = merge_reprocessed_draft(old, _engine([_clip("x", 0.0, 2.0)]))
    assert [c["clip_id"] for c in merged["selected"]] == ["x", "broll_abcd1234"]
    plain = merge_reprocessed_draft(_engine([_clip("a", 0.0, 2.0)]), _engine([_clip("x", 0.0, 2.0)]))
    assert [c["clip_id"] for c in plain["selected"]] == ["x"]
    assert plain["diagnostics"]["reprocess_merge"]["broll_reinserted"] == []


def test_store_reprocess_of_an_edited_draft_keeps_the_broll_clip_where_it_is():
    from cutsell_worker import draft_store
    from tests.test_cutsell_clean_worker_draft_store import FakeRedis

    client = FakeRedis()
    sources = [{"source_asset_id": "s1", "uri": "s3://b/s.mov"}]
    draft_store.create_initial_draft(user_id="u", project_id="p1", draft=_engine([_clip("a", 0.0, 4.0)]),
                                     sources=sources, client=client)
    edited = _engine([_clip("a", 0.0, 2.0), _broll(), _clip("a2", 2.0, 4.0)], broll_sources=BROLL_SOURCES)
    draft_store.save_draft_snapshot(user_id="u", project_id="p1", draft=edited, expected_revision=1, client=client)
    stored, _ = draft_store.create_initial_draft(user_id="u", project_id="p1",
                                                 draft=_engine([_clip("x", 0.0, 4.0)]), sources=sources, client=client)
    assert [c["clip_id"] for c in stored["draft"]["selected"]] == ["a", "broll_abcd1234", "a2"]
    assert stored["draft"]["broll_sources"] == BROLL_SOURCES


# --- the same job running again on a draft the creator edited -------------------------------------

def _store():
    from cutsell_worker import draft_store
    from tests.test_cutsell_clean_worker_draft_store import FakeRedis
    return draft_store, FakeRedis()


SOURCES = [{"source_asset_id": "s1", "uri": "s3://b/s.mov"}]
AI_ZOOM = {"keyframe_id": "z_ai", "source_asset_id": "s1", "source_time": 1.0, "duration": 1.0, "scale": 1.1, "origin": "ai"}


def _engine_cut(**extra):
    return _engine([_clip("a", 0.0, 3.0, text="one two", words=_words(("one", 0.5, 1.0), ("two", 2.0, 2.5))),
                    _clip("b", 4.0, 6.0), _clip("c", 7.0, 9.0)],
                   zoom_keyframes=[AI_ZOOM], auto_zoom_enabled=True, **extra)


def test_job_retry_keeps_deleted_split_and_reordered_clips():
    draft_store, client = _store()
    engine = _engine_cut()
    draft_store.create_initial_draft(user_id="u", project_id="p1", draft=engine, sources=SOURCES, client=client)
    # The creator deletes "b", splits "a" at 1.5 s and moves "c" first.
    a_left = _clip("a", 0.0, 1.5, text="one", words=_words(("one", 0.5, 1.0)))
    a_right = _clip("a-r", 1.5, 3.0, text="two", words=_words(("two", 2.0, 2.5)))
    creator = copy.deepcopy(engine)
    creator["selected"] = [_clip("c", 7.0, 9.0), a_left, a_right]
    creator["discarded"] = [_clip("b", 4.0, 6.0)]
    saved = draft_store.save_draft_snapshot(user_id="u", project_id="p1", draft=creator, expected_revision=1,
                                            client=client)
    # The same flow-b job runs again (worker restart) and produces the same engine draft.
    stored, created = draft_store.create_initial_draft(user_id="u", project_id="p1", draft=_engine_cut(),
                                                       sources=SOURCES, client=client)
    assert created is False
    assert stored["draft"]["selected"] == creator["selected"]
    assert stored["draft"]["discarded"] == creator["discarded"]
    assert stored["draft"]["alternates"] == creator["alternates"]
    # nothing of the AI's changed: no new revision was written
    assert stored["revision"] == saved["revision"] == 2


def test_job_retry_refreshes_only_the_ai_zooms_and_diagnostics():
    draft_store, client = _store()
    engine = _engine_cut()
    draft_store.create_initial_draft(user_id="u", project_id="p1", draft=engine, sources=SOURCES, client=client)
    creator = copy.deepcopy(engine)
    creator["selected"] = list(reversed(creator["selected"]))
    creator["zoom_keyframes"] = [AI_ZOOM, {"keyframe_id": "mine", "source_asset_id": "s1", "source_time": 7.5,
                                           "duration": 1.0, "scale": 1.2, "origin": "user"},
                                 {"keyframe_id": "z_ai2", "source_asset_id": "s1", "source_time": 4.5,
                                  "duration": 1.0, "scale": 1.1, "origin": "ai"}]
    creator["zoom_removed"] = [{"source_asset_id": "s1", "source_time": 2.5}]
    draft_store.save_draft_snapshot(user_id="u", project_id="p1", draft=creator, expected_revision=1, client=client)
    rerun = _engine_cut(diagnostics={"engine": "simple", "run": 2})
    rerun["zoom_keyframes"] = [
        {"keyframe_id": "n1", "source_asset_id": "s1", "source_time": 5.0, "duration": 1.0, "scale": 1.1, "origin": "ai"},
        {"keyframe_id": "n2", "source_asset_id": "s1", "source_time": 2.6, "duration": 1.0, "scale": 1.1, "origin": "ai"},
    ]
    stored, _ = draft_store.create_initial_draft(user_id="u", project_id="p1", draft=rerun, sources=SOURCES, client=client)
    draft = stored["draft"]
    assert stored["revision"] == 3 and stored["undo"][-1] == creator
    assert draft["selected"] == creator["selected"]
    assert [k["keyframe_id"] for k in draft["zoom_keyframes"]] == ["mine", "n1"]  # n2 is tombstoned
    assert draft["diagnostics"]["run"] == 2
    assert draft["diagnostics"]["reprocess_merge"]["kept_creator_cut"] is True
    for key in ("zoom_removed", "auto_zoom_enabled", "caption_preset", "captions_enabled"):
        assert draft[key] == creator[key]
    draft_from_dict(draft)


def test_untouched_and_first_run_paths_are_unchanged():
    draft_store, client = _store()
    first, created = draft_store.create_initial_draft(user_id="u", project_id="p1", draft=_engine_cut(),
                                                      sources=SOURCES, client=client)
    assert created and first["draft"] == _engine_cut()                 # first run: stored as is
    new = _engine([_clip("x", 0.0, 2.0)], zoom_keyframes=[AI_ZOOM])
    untouched, _ = draft_store.create_initial_draft(user_id="u", project_id="p1", draft=new, sources=SOURCES,
                                                    client=client)
    assert untouched["revision"] == 2
    assert untouched["draft"] == merge_reprocessed_draft(first["draft"], new)   # untouched: new cut
    assert [c["clip_id"] for c in untouched["draft"]["selected"]] == ["x"]
    assert "kept_creator_cut" not in untouched["draft"]["diagnostics"]["reprocess_merge"]
