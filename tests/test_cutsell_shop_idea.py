"""Shop board "My idea" (PO 18 Nov 2026): one Sonnet look at her screenshot -> an exact map; nothing stored."""
import base64

import pytest
from fastapi.testclient import TestClient

from cutsell_app import shop_idea_routes as R

JPEG = base64.b64encode(b"\xff\xd8\xff\xe0" + b"0" * 400).decode()


def _call(answer, usage=None):
    return lambda body: (answer, usage or {"input_tokens": 1600, "output_tokens": 900}, None)


GOOD = {"background": {"top": "#fce0e5", "bottom": "nope"},
        "texts": [{"text": "glow kit", "role": "title", "x": 0.07, "y": 0.05, "w": 0.6, "h": 0.09, "color": "#9E2654",
                   "font": "display", "weight": "black", "align": "left", "uppercase": False, "rotation": 0},
                  {"text": "  ", "role": "other", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1, "color": "#000000",
                   "font": "sans", "weight": "regular", "align": "left", "uppercase": False, "rotation": 0}],
        "slots": [{"kind": "top", "x": 0.09, "y": 0.2, "w": 0.4, "h": 0.32, "rotation": -4, "front": 2},
                  {"kind": "spaceship", "x": 0.95, "y": 0.9, "w": 0.4, "h": 0.4, "rotation": 400, "front": 1},
                  {"kind": "shoes", "x": 0.2, "y": 0.6, "w": 0.001, "h": 0.2, "rotation": 0, "front": 3}]}


def test_the_map_is_checked_and_bounded():
    out = R.read_idea(JPEG, call=_call(GOOD))
    style = out["style"]
    assert style["background"] == {"top": "#FCE0E5", "bottom": "#FCE0E5"}          # a broken colour = the other one
    assert [t["text"] for t in style["texts"]] == ["glow kit"]                     # empty text dropped
    assert [s["kind"] for s in style["slots"]] == ["top", "other"]                 # too thin dropped; unknown = other
    odd = style["slots"][1]
    assert odd["x"] + odd["w"] <= 1 and odd["y"] + odd["h"] <= 1 and odd["rotation"] == 180
    assert out["cost_usd"] == pytest.approx((1600 * 2 + 900 * 10) / 1e6)


def test_a_picture_with_no_products_is_refused_and_bad_files_never_reach_the_ai():
    assert R.read_idea(JPEG, call=_call({"background": {}, "texts": [], "slots": []}))["error"] == "no_products_found"
    called = []
    out = R.read_idea(base64.b64encode(b"GIF89a" + b"0" * 200).decode(), call=lambda b: called.append(b))
    assert out["error"] == "bad_image" and not called


def test_one_call_with_the_picture_and_the_exact_copy_prompt():
    seen = {}
    R.read_idea(JPEG, call=lambda body: (seen.update(body), (GOOD, {}, None))[1])
    content = seen["messages"][0]["content"]
    assert seen["model"] == "claude-sonnet-5-5" and content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg" and "EXACTLY" in content[1]["text"]


def test_route_daily_limit_and_nothing_logged(monkeypatch, caplog):
    from cutsell_app.main import app
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "0")
    monkeypatch.setattr(R, "read_idea", lambda image: {"style": {"background": {}, "texts": [], "slots": [{}]},
                                                       "error": None, "cost_usd": 0.01})
    monkeypatch.setattr(R, "DAILY_LIMIT", 2)
    R._COUNTS.clear()
    client = TestClient(app)
    body = {"user_id": "u-idea", "image_b64": JPEG}
    assert client.post("/v1/shop/idea", json=body).status_code == 200
    assert client.post("/v1/shop/idea", json=body).status_code == 200
    assert client.post("/v1/shop/idea", json=body).status_code == 429
    assert JPEG[:40] not in caplog.text
