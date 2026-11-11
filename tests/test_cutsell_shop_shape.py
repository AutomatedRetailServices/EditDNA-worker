"""Shop board "Shape it" (POST /v1/shop/shape): auth, switch, limits, providers, errors, costs. Mocks only --
no network, no real image model, synthetic PNGs made in memory."""
from __future__ import annotations

import base64
import io

import fakeredis
import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

import cutsell_app.auth_middleware as middleware
from cutsell_app import shop_shape as shape
from cutsell_app.main import app
from cutsell_worker.auth import user_apple_key

USER = "usr_apple_shopper"
GUEST = "usr_guest123"
TOKENS = {"tok-user": USER, "tok-guest": GUEST}
AUTH = {"Authorization": "Bearer tok-user"}
DAY0 = 1_794_000_000.0          # a fixed UTC instant (Nov 2026)


def garment_png(width=400, height=500, colour=(30, 60, 160, 255), alpha=True) -> bytes:
    image = Image.new("RGBA" if alpha else "RGB", (width, height), (0, 0, 0, 0) if alpha else (255, 255, 255))
    for x in range(width // 4, 3 * width // 4):
        for y in range(height // 5, 4 * height // 5):
            image.putpixel((x, y), colour if alpha else colour[:3])
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def opaque_output(size=(1024, 1024), bg=(255, 255, 255), fg=(30, 60, 160)) -> bytes:
    image = Image.new("RGB", size, bg)
    w, h = size
    image.paste(Image.new("RGB", (w // 2, h // 2), fg), (w // 4, h // 4))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def alpha_output(size=(1024, 1024)) -> bytes:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    w, h = size
    image.paste(Image.new("RGBA", (w // 2, h // 2), (30, 60, 160, 255)), (w // 4, h // 4))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class FakeResponse:
    def __init__(self, status: int, data: dict | None = None):
        self.status_code = status
        self._data = data or {}

    def json(self):
        return self._data


class FakeHTTP:
    """Records every POST; answers from a queue (one entry per call)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}), "json": json, "timeout": timeout})
        answer = self.answers.pop(0) if self.answers else FakeResponse(500)
        if isinstance(answer, Exception):
            raise answer
        return answer


def seedream_ok():
    return FakeResponse(200, {"data": [{"b64_json": b64(opaque_output())}], "usage": {"generated_images": 1}})


def openai_ok():
    return FakeResponse(200, {"data": [{"b64_json": b64(alpha_output())}],
                              "usage": {"input_tokens": 1500, "output_tokens": 1756,
                                        "input_tokens_details": {"image_tokens": 1200, "text_tokens": 300}}})


def gemini_ok():
    return FakeResponse(200, {"candidates": [{"content": {"parts": [
        {"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": b64(opaque_output())}}]}}],
        "usageMetadata": {"promptTokenCount": 1600, "candidatesTokenCount": 1130,
                          "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120},
                                                      {"modality": "TEXT", "tokenCount": 10}]}})


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")
    monkeypatch.setenv("CUTSELL_SHAPE_ENABLED", "1")
    monkeypatch.setenv("CUTSELL_SHAPE_MODEL", "seedream")
    monkeypatch.setenv("CUTSELL_SEEDREAM_API_KEY", "test-seedream-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    for name in ("CUTSELL_SHAPE_DAILY_LIMIT", "CUTSELL_SHAPE_MAX_BYTES", "CUTSELL_SHAPE_MAX_SIDE"):
        monkeypatch.delenv(name, raising=False)

    def resolve(token):
        if token in TOKENS:
            return {"user_id": TOKENS[token]}
        raise PermissionError("invalid or expired bearer token")

    monkeypatch.setattr(middleware, "resolve_session", resolve)
    redis = fakeredis.FakeRedis()
    redis.set(user_apple_key(USER), "apple-scope")
    monkeypatch.setattr(shape, "_redis", lambda: redis)
    clock = {"now": DAY0}
    monkeypatch.setattr(shape, "_now", lambda: clock["now"])
    monkeypatch.setattr(shape.time, "sleep", lambda s: None)
    events = []
    import cutsell_worker.commercial_observability as obs
    monkeypatch.setattr(obs, "capture_operational_event", lambda name, **d: events.append((name, d)))
    http = FakeHTTP()
    monkeypatch.setattr(shape.requests, "post", http)
    return {"redis": redis, "clock": clock, "events": events, "http": http, "client": TestClient(app)}


def body(kind="clothing", png=None):
    return {"image_png_base64": b64(png or garment_png()), "kind": kind}


# ------------------------------------------------------------------------------------------------ auth + switch
def test_signed_out_is_401(env):
    assert env["client"].post("/v1/shop/shape", json=body()).status_code == 401
    assert env["client"].post("/v1/shop/shape", json=body(), headers={"Authorization": "Bearer nope"}).status_code == 401
    assert env["http"].calls == []


def test_guest_or_probe_account_is_401(env):
    r = env["client"].post("/v1/shop/shape", json=body(), headers={"Authorization": "Bearer tok-guest"})
    assert r.status_code == 401 and r.json() == {"error": "sign_in_required"}
    assert env["http"].calls == []


def test_guest_upgraded_by_apple_counts_as_signed_in(env):
    env["redis"].set(user_apple_key(GUEST), "apple-scope")
    env["http"].answers.append(seedream_ok())
    r = env["client"].post("/v1/shop/shape", json=body(), headers={"Authorization": "Bearer tok-guest"})
    assert r.status_code == 200


def test_disabled_is_503(env, monkeypatch):
    monkeypatch.setenv("CUTSELL_SHAPE_ENABLED", "0")
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 503 and r.json() == {"error": "shape_disabled"}
    assert env["http"].calls == []


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CUTSELL_SHAPE_ENABLED", raising=False)
    assert shape.shape_enabled() is False


def test_redis_down_fails_closed(env, monkeypatch):
    def broken():
        raise RuntimeError("REDIS_URL is not configured")
    monkeypatch.setattr(shape, "_redis", broken)
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 503 and r.json() == {"error": "shape_meter_unavailable"}
    assert env["http"].calls == []


def test_counter_error_fails_closed(env, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(env["redis"], "incr", boom)
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 503 and r.json()["error"] == "shape_meter_unavailable"
    assert env["http"].calls == []


# ------------------------------------------------------------------------------------------------ validation
@pytest.mark.parametrize("kind", ["hat", "", None, 3, ["shoes"]])
def test_bad_kind_is_400(env, kind):
    r = env["client"].post("/v1/shop/shape", json={"image_png_base64": b64(garment_png()), "kind": kind},
                           headers=AUTH)
    assert r.status_code == 400 and r.json() == {"error": "bad_kind"}
    assert env["http"].calls == []


@pytest.mark.parametrize("value,code", [
    ("not base64 !!", "bad_image"),
    (b64(b"GIF89a....."), "bad_image"),
    (b64(garment_png(alpha=False)), "image_needs_transparency"),
    (b64(Image.new("RGBA", (64, 64), (0, 0, 0, 0)).tobytes()), "bad_image"),
    (None, "bad_image"),
])
def test_bad_image_is_400(env, value, code):
    r = env["client"].post("/v1/shop/shape", json={"image_png_base64": value, "kind": "clothing"}, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"] == code
    assert env["http"].calls == []


def test_fully_transparent_image_is_400(env):
    empty = io.BytesIO()
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(empty, format="PNG")
    r = env["client"].post("/v1/shop/shape", json=body(png=empty.getvalue()), headers=AUTH)
    assert r.status_code == 400


def test_invalid_json_is_400(env):
    r = env["client"].post("/v1/shop/shape", content=b"{nope", headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 400


def test_too_many_pixels_on_a_side_is_413(env, monkeypatch):
    monkeypatch.setenv("CUTSELL_SHAPE_MAX_SIDE", "300")
    r = env["client"].post("/v1/shop/shape", json=body(png=garment_png(400, 200)), headers=AUTH)
    assert r.status_code == 413 and r.json()["error"] == "image_too_large" and r.json()["max_side"] == 300
    assert env["http"].calls == []


def test_too_many_bytes_is_413(env, monkeypatch):
    png = garment_png()
    monkeypatch.setenv("CUTSELL_SHAPE_MAX_BYTES", str(len(png) - 100))
    r = env["client"].post("/v1/shop/shape", json=body(png=png), headers=AUTH)
    assert r.status_code == 413 and r.json()["max_bytes"] == len(png) - 100


def test_default_limits():
    assert shape.DEFAULT_MAX_SIDE == 2048 and shape.DEFAULT_DAILY_LIMIT == 20
    # the default must fit through the auth middleware's JSON body cap once base64-encoded
    assert shape.DEFAULT_MAX_BYTES * 4 / 3 + 1024 < middleware.MAX_JSON_BODY


# ------------------------------------------------------------------------------------------------ daily limit
def test_daily_limit_21st_is_429_and_resets_next_utc_day(env):
    for _ in range(20):
        env["http"].answers.append(seedream_ok())
        assert env["client"].post("/v1/shop/shape", json=body(), headers=AUTH).status_code == 200
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 429 and r.json() == {"error": "shape_daily_limit", "limit": 20}
    assert len(env["http"].calls) == 20                 # the 21st never reached the provider
    env["clock"]["now"] = DAY0 + 86400                  # next UTC day
    env["http"].answers.append(seedream_ok())
    assert env["client"].post("/v1/shop/shape", json=body(), headers=AUTH).status_code == 200


def test_limit_is_per_account_and_hashed(env):
    env["http"].answers.append(seedream_ok())
    env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    keys = [k.decode() for k in env["redis"].keys("cutsell:v1:shape-day:*")]
    assert len(keys) == 1 and USER not in keys[0]
    assert int(env["redis"].get(keys[0])) == 1


def test_failed_call_does_not_use_up_a_slot(env):
    env["http"].answers.extend([FakeResponse(500), FakeResponse(500)])
    assert env["client"].post("/v1/shop/shape", json=body(), headers=AUTH).status_code == 502
    assert int(env["redis"].get(shape.day_key(USER)) or 0) == 0


# ------------------------------------------------------------------------------------------------ providers
def test_seedream_request_shape(env):
    env["http"].answers.append(seedream_ok())
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 200 and r.json()["provider"] == "seedream"
    call = env["http"].calls[0]
    assert call["url"] == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    assert call["headers"]["Authorization"] == "Bearer test-seedream-key"
    j = call["json"]
    assert j["model"] == "seedream-5-0-flash-260915" and j["response_format"] == "b64_json"
    assert j["watermark"] is False and j["output_format"] == "png"
    assert j["image"].startswith("data:image/png;base64,")
    assert "ghost mannequin" in j["prompt"] and "pure white" in j["prompt"]
    assert call["timeout"] == (shape.CONNECT_TIMEOUT, shape.READ_TIMEOUT)


def test_openai_request_shape(env, monkeypatch):
    monkeypatch.setenv("CUTSELL_SHAPE_MODEL", "gpt-image")
    env["http"].answers.append(openai_ok())
    r = env["client"].post("/v1/shop/shape", json=body("shoes"), headers=AUTH)
    assert r.status_code == 200 and r.json()["provider"] == "gpt-image"
    call = env["http"].calls[0]
    assert call["url"] == "https://api.openai.com/v1/images/edits"
    assert call["headers"]["Authorization"] == "Bearer test-openai-key"
    j = call["json"]
    assert j["model"] == "gpt-image-2.5-flare" and j["background"] == "transparent"
    assert j["output_format"] == "png" and j["n"] == 1 and j["size"] == "1024x1536"   # portrait input
    assert j["images"][0]["image_url"].startswith("data:image/png;base64,")
    assert "shoes" in j["prompt"] and "transparent" in j["prompt"]
    # cost from usage: 300 text in, 1200 image in, 1756 out
    assert r.json()["cost_usd"] == pytest.approx((300 * 5 + 1200 * 8 + 1756 * 30) / 1e6)


def test_gemini_request_shape(env, monkeypatch):
    monkeypatch.setenv("CUTSELL_SHAPE_MODEL", "nano-banana")
    env["http"].answers.append(gemini_ok())
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 200 and r.json()["provider"] == "nano-banana"
    call = env["http"].calls[0]
    assert call["url"] == ("https://generativelanguage.googleapis.com/v1/models/"
                           "gemini-3.1-flash-lite-image:generateContent")
    assert call["headers"]["x-goog-api-key"] == "test-gemini-key"
    parts = call["json"]["contents"][0]["parts"]
    assert "ghost mannequin" in parts[0]["text"]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert call["json"]["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
    assert r.json()["cost_usd"] == pytest.approx((1600 * 0.25 + 1120 * 30 + 10 * 1.5) / 1e6)


@pytest.mark.parametrize("value,name", [("seedream", "seedream"), ("gpt-image-2.5", "gpt-image"),
                                        ("nano-banana-2-lite", "nano-banana"), ("GEMINI", "nano-banana")])
def test_provider_aliases(value, name):
    assert shape.provider_name(value) == name


def test_unknown_model_is_503(env, monkeypatch):
    monkeypatch.setenv("CUTSELL_SHAPE_MODEL", "dall-e-9")
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 503 and r.json()["error"] == "shape_provider_not_configured"


def test_missing_key_is_503_and_no_call(env, monkeypatch):
    monkeypatch.delenv("CUTSELL_SEEDREAM_API_KEY")
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 503 and r.json()["error"] == "shape_provider_not_configured"
    assert env["http"].calls == []
    assert int(env["redis"].get(shape.day_key(USER)) or 0) == 0


# ------------------------------------------------------------------------------------------------ provider errors
def test_provider_error_is_502_with_safe_message(env):
    env["http"].answers.extend([FakeResponse(500, {"error": {"message": "secret internal detail"}}),
                                FakeResponse(503)])
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 502 and r.json() == {"error": "shape_provider_error", "provider": "seedream"}
    assert "secret" not in r.text
    assert len(env["http"].calls) == 2                  # one retry, no more


def test_4xx_is_not_retried(env):
    env["http"].answers.extend([FakeResponse(400), seedream_ok()])
    assert env["client"].post("/v1/shop/shape", json=body(), headers=AUTH).status_code == 502
    assert len(env["http"].calls) == 1


def test_network_error_retried_once_then_ok(env):
    env["http"].answers.extend([requests.ConnectionError("x"), seedream_ok()])
    assert env["client"].post("/v1/shop/shape", json=body(), headers=AUTH).status_code == 200
    assert len(env["http"].calls) == 2


def test_no_image_in_answer_is_502(env, monkeypatch):
    monkeypatch.setenv("CUTSELL_SHAPE_MODEL", "nano-banana")
    env["http"].answers.append(FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "no"}]}}]}))
    r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
    assert r.status_code == 502 and r.json() == {"error": "shape_no_image", "provider": "nano-banana"}


# ------------------------------------------------------------------------------------------------ cost + output
def test_cost_is_recorded_and_returned(env):
    env["http"].answers.extend([seedream_ok(), seedream_ok()])
    for _ in range(2):
        r = env["client"].post("/v1/shop/shape", json=body(), headers=AUTH)
        assert r.json()["cost_usd"] == pytest.approx(0.018)
    ledger = {k.decode(): float(v) for k, v in env["redis"].hgetall(shape.cost_key()).items()}
    assert ledger["total_usd"] == pytest.approx(0.036) and ledger["seedream_usd"] == pytest.approx(0.036)
    assert ledger["seedream_calls"] == 2
    calls = [d for name, d in env["events"] if name == "shop.shape.call"]
    assert len(calls) == 2 and calls[0]["cost_usd"] == pytest.approx(0.018) and calls[0]["status"] == "ok"
    assert USER not in str(calls) and "test-seedream-key" not in str(calls)
    assert all(len(str(d)) < 400 for d in calls)       # no image bytes in the log line


def test_output_is_transparent_png_on_input_canvas(env):
    env["http"].answers.append(seedream_ok())
    r = env["client"].post("/v1/shop/shape", json=body(png=garment_png(400, 500)), headers=AUTH)
    out = Image.open(io.BytesIO(base64.b64decode(r.json()["image_png_base64"])))
    assert out.format == "PNG" and out.mode == "RGBA" and out.size == (400, 500)
    assert out.getpixel((2, 2))[3] == 0                  # keyed-out background
    assert out.getpixel((200, 250))[3] == 255            # garment stays opaque


def test_key_out_keeps_key_coloured_area_inside_garment():
    image = Image.new("RGB", (300, 300), (255, 255, 255))
    image.paste(Image.new("RGB", (200, 200), (20, 20, 120)), (50, 50))
    image.paste(Image.new("RGB", (40, 40), (255, 255, 255)), (130, 130))   # white logo inside the garment
    keyed = shape._key_out(image, (255, 255, 255))
    assert keyed.getpixel((5, 5))[3] == 0
    assert keyed.getpixel((150, 150))[3] == 255


def test_light_garment_gets_dark_key():
    garment = shape.decode_image(b64(garment_png(colour=(240, 240, 235, 255))))
    assert garment.key_rgb == (24, 24, 24)
    assert shape.decode_image(b64(garment_png())).key_rgb == (255, 255, 255)
