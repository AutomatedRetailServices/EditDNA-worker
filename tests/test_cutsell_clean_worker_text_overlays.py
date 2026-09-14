from pathlib import Path

from fastapi.testclient import TestClient

import cutsell_app.main as api
import cutsell_worker.render as renderer
from cutsell_worker.contracts import TextOverlay
from cutsell_worker.serde import draft_from_dict
from cutsell_worker.text_edits import add_text_overlay, remove_text_overlay, update_text_overlay


def _draft():
    return {
        "schema_version": "cutsell.v1",
        "project_id": "p1",
        "strategy": "mixed",
        "selected": [{
            "clip_id": "c1", "source_asset_id": "s1", "source_order": 0,
            "start": 0.0, "end": 4.0, "text": "hello", "caption_text": "hello", "selected": True,
        }],
        "alternates": [], "discarded": [], "diagnostics": {},
    }


def test_add_update_remove_text_overlay_are_stateless_and_bounded():
    original = _draft()
    added = add_text_overlay(original, text="SALE", start=0.5, end=2.0, x=0.7, y=0.25, scale=1.4)
    assert len(added["text_overlays"]) == 1
    overlay_id = added["text_overlays"][0]["overlay_id"]
    assert added["text_overlays"][0]["x"] == 0.7
    assert "text_overlays" not in original

    updated = update_text_overlay(added, overlay_id, text="50% OFF", x=0.5, scale=1.8)
    assert updated["text_overlays"][0]["text"] == "50% OFF"
    assert updated["text_overlays"][0]["scale"] == 1.8

    removed = remove_text_overlay(updated, overlay_id)
    assert removed["text_overlays"] == []

    try:
        add_text_overlay(original, text="bad", start=0.0, end=5.0)
    except ValueError as exc:
        assert "inside draft duration" in str(exc)
    else:
        raise AssertionError("overlay beyond draft duration must be rejected")


def test_mobile_text_routes_cover_add_update_remove():
    client = TestClient(api.app)
    added = client.post("/v1/draft-edits/text/add", json={
        "draft": _draft(), "text": "SALE", "start": 0.5, "end": 2.0, "x": 0.5, "y": 0.2, "scale": 1.0,
    })
    assert added.status_code == 200
    draft = added.json()
    overlay_id = draft["text_overlays"][0]["overlay_id"]

    updated = client.post("/v1/draft-edits/text/update", json={
        "draft": draft, "overlay_id": overlay_id, "text": "BUY NOW", "y": 0.8,
    })
    assert updated.status_code == 200
    assert updated.json()["text_overlays"][0]["text"] == "BUY NOW"
    assert updated.json()["text_overlays"][0]["y"] == 0.8

    removed = client.post("/v1/draft-edits/text/remove", json={"draft": updated.json(), "overlay_id": overlay_id})
    assert removed.status_code == 200
    assert removed.json()["text_overlays"] == []


def test_serde_rejects_invalid_overlay_and_preserves_valid_track():
    payload = _draft()
    payload["text_overlays"] = [{
        "overlay_id": "txt1", "text": "SALE", "start": 0.2, "end": 1.2, "x": 0.4, "y": 0.3, "scale": 1.2,
    }]
    parsed = draft_from_dict(payload)
    assert parsed.text_overlays[0].text == "SALE"
    assert parsed.text_overlays[0].scale == 1.2

    payload["text_overlays"][0]["x"] = 2.0
    try:
        draft_from_dict(payload)
    except ValueError as exc:
        assert "normalized" in str(exc)
    else:
        raise AssertionError("invalid overlay position must be rejected")


def test_ass_overlay_file_contains_global_timing_position_and_size(tmp_path):
    ass = tmp_path / "text.ass"
    renderer._write_text_overlay_ass(
        (TextOverlay("txt1", "SALE", 1.25, 3.5, x=0.75, y=0.25, scale=1.5),),
        ass,
        width=1080,
        height=1920,
    )
    content = ass.read_text(encoding="utf-8")
    assert "Dialogue: 0,0:00:01.25,0:00:03.50" in content
    assert r"\pos(810,480)" in content
    assert r"\fs72" in content
    assert "SALE" in content
