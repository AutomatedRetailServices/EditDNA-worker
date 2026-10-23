"""23 Oct (PO): a video that arrives with the brand / product the creator chose skips the Haiku detection."""
from __future__ import annotations

from types import SimpleNamespace

import fakeredis
import pytest

from cutsell_worker import product_catalog, product_tagging


@pytest.fixture
def r(monkeypatch):
    client = fakeredis.FakeRedis()
    monkeypatch.setattr(product_catalog, "_redis", lambda client_=None, **_: client, raising=False)
    return client


def _src(**meta):
    return [SimpleNamespace(metadata=meta)]


def _no_haiku(monkeypatch):
    calls = []
    monkeypatch.setattr(product_tagging, "tag", lambda *a, **k: calls.append(1) or {"brand": "X", "product": "Y",
                                                                                     "confidence": 0.9})
    return calls


def test_given_names_skip_detection_and_place(r, monkeypatch):
    calls = _no_haiku(monkeypatch)
    out = product_tagging.tag_and_record(user_id="u", project_id="p1", draft={}, sources=_src(brand="CeraVe",
                                         product="Hydrating Cleanser"), video_path=None, client=r)
    assert out["status"] == "given" and out["cost_usd"] == 0.0 and not calls
    entry = product_catalog.video_entry("u", "p1", client=r)
    assert entry["source"] == "user" and entry["product_id"] == out["product_id"]


def test_given_id_and_unknown_id_falls_back_to_names(r, monkeypatch):
    calls = _no_haiku(monkeypatch)
    made = product_catalog.create("u", brand="Dior", product="Lip Glow", client=r)
    out = product_tagging.tag_and_record(user_id="u", project_id="p1", draft={},
                                         sources=_src(product_id=made["product_id"]), video_path=None, client=r)
    assert out["product_id"] == made["product_id"]
    out2 = product_tagging.tag_and_record(user_id="u", project_id="p2", draft={},
                                          sources=_src(product_id="nope", brand="Dior", product="Lip Glow"),
                                          video_path=None, client=r)
    assert out2["product_id"] == made["product_id"] and not calls


def test_assigned_by_hand_before_the_cut_skips(r, monkeypatch):
    calls = _no_haiku(monkeypatch)
    product_catalog.assign("u", ["p1"], brand="CeraVe", product="Cream", client=r)
    out = product_tagging.tag_and_record(user_id="u", project_id="p1", draft={}, sources=_src(), video_path=None, client=r)
    assert out["status"] == "given" and not calls


def test_no_product_still_detects(r, monkeypatch):
    calls = _no_haiku(monkeypatch)
    out = product_tagging.tag_and_record(user_id="u", project_id="p9", draft={}, sources=_src(album_hint="x"),
                                         video_path=None, client=r)
    assert out["status"] == "done" and calls == [1]
