"""Shape it through kie.ai (PO 16 Nov 2026): the kie.ai client (Modal side) and the API's relay provider."""
from __future__ import annotations

import base64
import io
import json

import pytest

from cutsell_app import shop_shape as shape
from cutsell_worker import kie_image


class FakeResponse:
    def __init__(self, status=200, data=None, content=b""):
        self.status_code, self._data, self.content = status, data, content

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakeKie:
    """upload -> createTask -> recordInfo (generating, then `final`) -> result download."""

    def __init__(self, final="success", result=b"\x89PNGresult"):
        self.calls, self.final, self.result, self.polls = [], final, result, 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        assert headers["Authorization"] == "Bearer k"
        if url == kie_image.UPLOAD_URL:
            assert json["base64Data"].startswith("data:image/png;base64,")
            return FakeResponse(data={"code": 200, "data": {"downloadUrl": "https://kie.example/in.png"}})
        assert url == kie_image.CREATE_URL
        assert json["model"] == "google/nano-banana-edit"
        assert json["input"]["image_urls"] == ["https://kie.example/in.png"]
        return FakeResponse(data={"code": 200, "data": {"taskId": "t1"}})

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        if url == kie_image.RECORD_URL:
            self.polls += 1
            if self.polls == 1:
                return FakeResponse(data={"code": 200, "data": {"state": "generating"}})
            if self.final == "fail":
                return FakeResponse(data={"code": 200, "data": {"state": "fail", "failMsg": "nope"}})
            return FakeResponse(data={"code": 200, "data": {"state": "success",
                                                            "resultJson": json.dumps({"resultUrls": ["https://kie.example/out.png"]})}})
        return FakeResponse(content=self.result)


def test_kie_flow_uploads_creates_polls_and_downloads():
    fake = FakeKie()
    image, info = kie_image.shape_png(b"\x89PNGin", "prompt", key="k", session=fake, sleep=lambda s: None)
    assert image == b"\x89PNGresult"
    assert info["model"] == "google/nano-banana-edit" and info["cost_usd"] == 0.02
    assert [c[0] for c in fake.calls] == ["POST", "POST", "GET", "GET", "GET"]


def test_kie_failed_task_is_no_image_and_missing_key_is_not_configured():
    with pytest.raises(kie_image.KieError) as failed:
        kie_image.shape_png(b"\x89PNGin", "p", key="k", session=FakeKie(final="fail"), sleep=lambda s: None)
    assert failed.value.code == "shape_no_image"
    with pytest.raises(kie_image.KieError) as missing:
        kie_image.shape_png(b"\x89PNGin", "p", key="", session=FakeKie())
    assert missing.value.code == "shape_provider_not_configured"


def test_kie_gives_up_after_the_total_wait():
    class Never(FakeKie):
        def get(self, url, headers=None, params=None, timeout=None):
            return FakeResponse(data={"code": 200, "data": {"state": "generating"}})
    now = [0.0]

    def clock():
        now[0] += 30
        return now[0]
    with pytest.raises(kie_image.KieError) as late:
        kie_image.shape_png(b"\x89PNGin", "p", key="k", session=Never(), sleep=lambda s: None, clock=clock)
    assert "timed out" in late.value.detail


def _garment():
    from PIL import Image
    img = Image.new("RGBA", (64, 96), (0, 0, 0, 0))
    for x in range(16, 48):
        for y in range(16, 80):
            img.putpixel((x, y), (40, 60, 160, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return shape.decode_image(base64.b64encode(buf.getvalue()).decode("ascii"))


def test_kie_is_the_default_and_the_relay_url_comes_from_the_wake_url(monkeypatch):
    monkeypatch.delenv("CUTSELL_SHAPE_MODEL", raising=False)
    monkeypatch.delenv("CUTSELL_SHAPE_KIE_URL", raising=False)
    assert shape.provider_name() == "kie"
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_URL", "https://acme--cutsell-worker-wake.modal.run")
    assert shape.KieProvider.relay_url() == "https://acme--cutsell-worker-shape-kie.modal.run"
    monkeypatch.setenv("CUTSELL_SHAPE_KIE_URL", "https://x.example/shape")
    assert shape.KieProvider.relay_url() == "https://x.example/shape"


def test_kie_provider_posts_once_with_the_token_and_reads_the_image(monkeypatch):
    monkeypatch.setenv("CUTSELL_SHAPE_KIE_URL", "https://relay.example")
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_TOKEN", "tok")
    sent = []

    class Relay:
        def post(self, url, json=None, timeout=None):
            sent.append((url, json))
            return FakeResponse(data={"image_png_base64": base64.b64encode(b"\x89PNGx").decode(), "cost_usd": 0.02,
                                      "model": "google/nano-banana-edit", "ms": 9000})
    provider = shape.KieProvider(session=Relay())
    result = shape.call_provider(provider, _garment(), "clothing")
    assert result.image_bytes == b"\x89PNGx" and result.cost_usd == 0.02
    assert len(sent) == 1 and sent[0][0] == "https://relay.example"
    body = sent[0][1]
    assert body["token"] == "tok" and body["prompt"]
    assert base64.b64decode(body["image_png_base64"]).startswith(b"\x89PNG")


@pytest.mark.parametrize("status,error,expected", [
    (502, "shape_no_image", shape.ShapeProviderError),
    (502, "shape_provider_error", shape.ShapeProviderError),
    (503, "shape_provider_not_configured", shape.ShapeConfigError),
])
def test_kie_provider_maps_relay_errors(monkeypatch, status, error, expected):
    monkeypatch.setenv("CUTSELL_SHAPE_KIE_URL", "https://relay.example")
    monkeypatch.setenv("CUTSELL_WORKER_WAKE_TOKEN", "tok")

    class Relay:
        def post(self, url, json=None, timeout=None):
            return FakeResponse(status=status, data={"error": error})
    with pytest.raises(expected):
        shape.call_provider(shape.KieProvider(session=Relay()), _garment(), "clothing")
