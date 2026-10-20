"""Edits · Products: brands/products store, the Haiku step after the cut, and the API."""
import base64
import copy
import json
import subprocess

import fakeredis
import pytest

from cutsell_worker import product_catalog as catalog
from cutsell_worker import product_tagging as tagging


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


def test_names_are_normalized_so_variants_join():
    assert catalog.normalize("CeraVe®") == catalog.normalize("  cerave ") == "cerave"
    assert catalog.normalize("Miss  Dior™") == "miss dior"
    assert catalog.clean_name(" CeraVe®  Hydrating  Cleanser ") == "CeraVe Hydrating Cleanser"


def test_confident_ai_answer_places_the_video_and_reuses_names(r):
    catalog.record_ai_result("u", "p1", {"brand": "CeraVe", "product": "Hydrating Cleanser", "confidence": 0.9}, client=r)
    catalog.record_ai_result("u", "p2", {"brand": "Cerave®", "product": "hydrating  cleanser", "confidence": 0.8}, client=r)
    data = catalog.load("u", client=r)
    assert len(data["brands"]) == 1 and len(data["products"]) == 1
    assert data["videos"]["p1"]["product_id"] == data["videos"]["p2"]["product_id"]
    assert list(data["brands"].values())[0]["name"] == "CeraVe"


def test_low_confidence_stays_uncategorized_with_the_suggestion(r):
    entry = catalog.record_ai_result("u", "p1", {"brand": "CeraVe", "product": "Foaming Cleanser", "confidence": 0.5},
                                     client=r)
    assert entry["product_id"] is None and entry["suggestion"]["product"] == "Foaming Cleanser"
    out = catalog.overview("u", [{"project_id": "p1", "state": "draft_ready", "title": "t"}], client=r)
    assert out["brands"] == [] and out["uncategorized"][0]["suggestion"]["brand"] == "CeraVe"


def test_the_creators_choice_always_wins(r):
    catalog.assign("u", ["p1"], brand="Dior", product="Lip Glow", client=r)
    entry = catalog.record_ai_result("u", "p1", {"brand": "CeraVe", "product": "Cream", "confidence": 0.99}, client=r)
    data = catalog.load("u", client=r)
    assert entry["source"] == "user"
    assert data["products"][entry["product_id"]]["name"] == "Lip Glow"
    assert entry["suggestion"]["brand"] == "CeraVe"          # kept only as a suggestion


def test_assign_several_rename_and_merge(r):
    made = catalog.create("u", brand="CeraVe", product="Hydrating Cleanser", client=r)
    other = catalog.create("u", brand="Cera Ve", product="Hydrating Cleanser", client=r)
    moved = catalog.assign("u", ["p1", "p2", "p3"], product_id=other["product_id"], client=r)
    assert moved["moved"] == 3
    merged = catalog.rename("u", kind="brand", item_id=other["brand_id"], name="cerave", client=r)
    assert merged["merged"] and merged["brand_id"] == made["brand_id"]
    data = catalog.load("u", client=r)
    assert len(data["brands"]) == 1 and len(data["products"]) == 1          # duplicate product merged too
    assert {v["product_id"] for v in data["videos"].values()} == {made["product_id"]}
    catalog.rename("u", kind="product", item_id=made["product_id"], name="Hydrating Facial Cleanser", client=r)
    assert catalog.load("u", client=r)["products"][made["product_id"]]["name"] == "Hydrating Facial Cleanser"


def test_overview_counts_ready_drafts_processing(r):
    pid = catalog.create("u", brand="Dior", product="Miss Dior", client=r)["product_id"]
    catalog.assign("u", ["a", "b", "c"], product_id=pid, client=r)
    projects = [{"project_id": "a", "state": "draft_ready"}, {"project_id": "b", "state": "processing"},
                {"project_id": "c", "state": "failed"}, {"project_id": "d", "state": "draft_ready"}]
    out = catalog.overview("u", projects, client=r)
    product = out["brands"][0]["products"][0]
    assert (product["ready"], product["processing"], product["drafts"], product["videos"]) == (1, 1, 1, 3)
    assert [u["project_id"] for u in out["uncategorized"]] == ["d"]
    detail = catalog.product_videos("u", pid, projects, client=r)
    assert [v["bucket"] for v in detail["videos"]] == ["ready", "processing", "drafts"]


# ------------------------------------------------------------------ the Haiku step

def _draft():
    w = lambda t, s: {"text": t, "start": s, "end": s + 0.3}
    return {"selected": [{"source_order": 0, "start": 0, "end": 5,
                          "words": [w("Hi", 0.1), w("look", 1.0), w("at", 1.3), w("this", 1.6), w("cleanser.", 2.0)]}],
            "alternates": [{"source_order": 0, "start": 5, "end": 6, "words": [w("otra", 5.1), w("vez", 5.4)]}]}


class _Resp:
    def __init__(self, data):
        self.status_code, self._data = 200, data

    def json(self):
        return self._data


def _answer(obj, tokens=(500, 40)):
    return _Resp({"content": [{"type": "thinking"}, {"type": "text", "text": json.dumps(obj)}],
                  "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}})


def test_transcript_has_every_word_with_times():
    text, first = tagging.transcript_lines(_draft())
    assert "[0.1s] Hi look at this cleanser." in text and "otra vez" in text
    assert first[3] == {"w": "this", "s": 1.6}


def test_text_is_enough_no_photo(monkeypatch):
    calls = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(tagging.requests, "post", lambda url, json=None, **k: calls.append(json) or _answer(
        {"brand": "CeraVe", "product": "Hydrating Cleanser", "category": "Skincare", "confidence": 0.92,
         "evidence": "cerave hydrating cleanser", "product_moment_seconds": 1.0}))
    out = tagging.tag(_draft(), [{"brand": "CeraVe", "product": "Hydrating Cleanser"}], video_path="/nope.mp4")
    assert out["brand"] == "CeraVe" and not out["used_photo"] and len(calls) == 1
    assert calls[0]["model"] == "claude-haiku-5-5" and "HER EXISTING NAMES" in calls[0]["messages"][0]["content"][0]["text"]
    assert out["log"]["cost_usd"] == pytest.approx((500 * 0.10 + 40 * 0.50) / 1e6)


def test_low_confidence_reads_two_frames_of_the_video(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=360x640:r=30:d=8",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    answers = iter([{"brand": None, "product": None, "category": None, "confidence": 0.2, "evidence": None,
                     "product_moment_seconds": None},
                    {"brand": "CeraVe", "product": "Foaming Cleanser", "category": "Skincare", "confidence": 0.85,
                     "evidence": "CeraVe FOAMING FACIAL CLEANSER", "product_moment_seconds": None}])
    sent = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(tagging.requests, "post", lambda url, json=None, **k: sent.append(json) or _answer(next(answers)))
    out = tagging.tag(_draft(), [], video_path=str(video), duration=8.0)
    images = [c for c in sent[1]["messages"][0]["content"] if c["type"] == "image"]
    assert len(images) == 2 and out["used_photo"] and out["product"] == "Foaming Cleanser"
    assert out["thumb_jpeg_b64"] and base64.b64decode(out["thumb_jpeg_b64"])[:2] == b"\xff\xd8"
    assert tagging.moment_for_photo({"product_moment_seconds": None}, [{"w": "this", "s": 1.6}], 8.0) == [2.1, 3.1]
    assert tagging.moment_for_photo({"product_moment_seconds": None}, [], 30.0) == [2.0, 6.0]


def test_voice_only_asks_for_the_photo_later(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(tagging.requests, "post", lambda *a, **k: _answer(
        {"brand": "Dior", "product": None, "category": None, "confidence": 0.4, "evidence": None,
         "product_moment_seconds": 3.0}))
    out = tagging.tag(_draft(), [], video_path=None)
    assert out["needs_photo"] and not out["used_photo"]


def test_haiku_failing_never_breaks_anything(monkeypatch, r):
    def boom(*a, **k):
        raise tagging.requests.ConnectionError("down")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(tagging.requests, "post", boom)
    monkeypatch.setattr(tagging.time, "sleep", lambda s: None)
    draft = _draft()
    before = copy.deepcopy(draft)
    out = tagging.tag_and_record(user_id="u", project_id="p1", draft=draft, sources=[], video_path=None, client=r)
    assert out["status"] == "done" and out["placed"] is False and draft == before
    assert catalog.load("u", client=r)["videos"]["p1"]["product_id"] is None


def test_album_name_is_a_free_hint():
    from types import SimpleNamespace as NS
    assert tagging.album_hint_of([NS(metadata={"album_hint": "CeraVe haul"})]) == "CeraVe haul"
    assert tagging.album_hint_of([NS(metadata={})]) is None


def test_the_cut_engine_is_untouched():
    """The Gold cut cannot change: the cut engine's code and prompts are byte-identical to the live engine (f7f5772)."""
    diff = subprocess.run(["git", "diff", "--stat", "f7f5772", "--", "cutsell_worker/simple_engine",
                           "cutsell_worker/simple_engine_adapter.py"], capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("git history not available")
    assert diff.stdout.strip() == ""


# ------------------------------------------------------------------ API

def test_api_catalog_assign_rename(monkeypatch, r):
    from fastapi.testclient import TestClient
    import cutsell_app.catalog_routes as routes
    import cutsell_app.main as api
    monkeypatch.setattr(catalog, "_redis_client", lambda client=None: r)
    projects = [{"project_id": "p1", "state": "draft_ready", "title": "a"}, {"project_id": "p2", "state": "draft_ready"},
                {"project_id": "p3", "state": "processing"}]
    monkeypatch.setattr(routes, "list_projects", lambda **k: projects)
    client = TestClient(api.app)
    moved = client.post("/v1/catalog/assign", json={"user_id": "u", "project_ids": ["p1", "p2", "p3"],
                                                    "brand": "CeraVe", "product": "Hydrating Cleanser"}).json()
    assert moved["moved"] == 3
    body = client.get("/v1/catalog", params={"user_id": "u"}).json()
    product = body["brands"][0]["products"][0]
    assert body["brands"][0]["name"] == "CeraVe" and (product["ready"], product["processing"]) == (2, 1)
    assert client.post("/v1/catalog/assign", json={"user_id": "u", "project_ids": ["nope"], "brand": "x",
                                                   "product": "y"}).status_code == 404
    renamed = client.post("/v1/catalog/rename", json={"user_id": "u", "kind": "product",
                                                      "id": product["product_id"], "name": "Foaming Cleanser"})
    assert renamed.status_code == 200
    detail = client.get(f"/v1/catalog/products/{product['product_id']}", params={"user_id": "u"}).json()
    assert detail["name"] == "Foaming Cleanser" and len(detail["videos"]) == 3
    assert client.get(f"/v1/catalog/products/{product['product_id']}/thumbnail", params={"user_id": "u"}).status_code == 404


def test_a_brand_not_read_on_the_label_is_dropped():
    assert tagging._read_on_label({"brand": "Nike", "evidence": "PRO NIKE printed on the waistband"})
    assert not tagging._read_on_label({"brand": "Essentia", "evidence": "16.9 FL OZ (1 PT 0.9 FL OZ) 500ml"})
