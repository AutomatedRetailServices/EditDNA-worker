from worker.editable_draft import build_editable_draft


def _clip(cid, start, end, *, keep=True, slot="OTHER", source_index=0):
    return {
        "id": cid,
        "source_index": source_index,
        "source_start": start,
        "source_end": end,
        "text": f"clip {cid}",
        "slot": slot,
        "semantic_score": 0.8,
        "visual_score": 0.7,
        "score": 0.8,
        "meta": {"keep": keep, "semantic_v2": {}},
    }


def test_editable_draft_separates_selected_alternates_and_discards():
    clips = [
        _clip("a", 0.0, 2.0, slot="HOOK"),
        _clip("b", 2.0, 4.0, slot="HOOK"),
        _clip("c", 4.0, 5.0, keep=False),
    ]
    draft = build_editable_draft(
        clips,
        ["a"],
        mode="human",
        clean_cut_discard_diagnostics=[{
            "diagnostic_id": "source_000:d:micro",
            "clip_id": "d",
            "source_index": 0,
            "source_start": 5.0,
            "source_end": 5.02,
            "text": "tiny fragment",
            "reason": "discarded_invalid_microfragment",
        }],
    )
    assert draft["schema_version"] == "v1"
    assert draft["selected_clip_ids"] == ["a"]
    assert [item["clip_id"] for item in draft["alternates"]] == ["b"]
    assert [item["clip_id"] for item in draft["discarded"]] == ["c"]
    assert [item["clip_id"] for item in draft["boundary_discards"]] == ["d"]
    assert draft["counts"] == {"selected": 1, "alternates": 1, "discarded": 2}


def test_unselected_valid_other_remains_an_alternate():
    draft = build_editable_draft(
        [_clip("valid-other", 0.0, 2.0, slot="OTHER")],
        [],
        mode="human",
    )
    assert draft["selected"] == []
    assert [item["clip_id"] for item in draft["alternates"]] == ["valid-other"]
    assert draft["discarded"] == []


def test_selected_order_follows_composer_order_without_mutating_candidates():
    clips = [_clip("a", 0.0, 1.0), _clip("b", 1.0, 2.0)]
    draft = build_editable_draft(clips, ["b", "a", "b"], mode="human")
    assert draft["selected_clip_ids"] == ["b", "a"]
    assert clips[0]["meta"]["keep"] is True
    assert clips[1]["meta"]["keep"] is True
