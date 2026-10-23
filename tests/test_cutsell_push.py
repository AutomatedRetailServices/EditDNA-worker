"""23 Oct: Apple push notifications (Single / Combine one per cut; Bulk ONE when the whole send is done; Mix one)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import fakeredis
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from cutsell_worker import push

TOKEN_A = "a" * 64
TOKEN_B = "b" * 64


@pytest.fixture
def r():
    return fakeredis.FakeRedis()


@pytest.fixture
def apns(monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    monkeypatch.setenv("APNS_KEY_P8", pem.replace("\n", "\\n"))      # pasted on one line also works
    monkeypatch.setenv("APNS_KEY_ID", "KEYID12345")
    monkeypatch.setenv("APNS_TEAM_ID", "TEAMID6789")
    push._JWT.clear()
    return key.public_key()


class Apple:
    def __init__(self, status=200, text=""):
        self.calls, self.status, self.text = [], status, text

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, json.loads(body)))
        return self.status, self.text


def _src(bulk_id=None, size=None):
    meta = {"bulk_id": bulk_id, "bulk_size": str(size)} if bulk_id else {}
    return [SimpleNamespace(metadata=meta)]


def test_not_configured_sends_nothing(r, monkeypatch):
    for k in ("APNS_KEY_P8", "APNS_KEY_ID", "APNS_TEAM_ID"):
        monkeypatch.delenv(k, raising=False)
    push.register(user_id="u", token=TOKEN_A, client=r)
    apple = Apple()
    out = push.cut_finished(user_id="u", project_id="p1", sources=_src(), ok=True, client=r, post=apple)
    assert out["reason"] == "not_configured" and not apple.calls
    push.mark_configured(client=r)
    assert push.active(client=r) is False


def test_register_normalises_and_rejects(r):
    push.register(user_id="u", token="<" + TOKEN_A.upper() + ">", language="es-MX", environment="development", client=r)
    info = push._tokens("u", client=r)
    assert list(info) == [TOKEN_A] and info[TOKEN_A]["lang"] == "es" and info[TOKEN_A]["env"] == "sandbox"
    with pytest.raises(ValueError):
        push.register(user_id="u", token="xyz", client=r)


def test_single_ready_push_is_signed_and_addressed(r, apns):
    push.register(user_id="u", token=TOKEN_A, client=r)
    apple = Apple()
    out = push.cut_finished(user_id="u", project_id="p1", sources=_src(), ok=True, client=r, post=apple)
    assert out["sent"] == 1
    url, headers, body = apple.calls[0]
    assert url == f"https://api.push.apple.com/3/device/{TOKEN_A}"
    assert headers["apns-topic"] == "ai.cutsell.app" and headers["apns-push-type"] == "alert"
    claims = jwt.decode(headers["authorization"].split()[1], apns, algorithms=["ES256"])
    assert claims["iss"] == "TEAMID6789"
    assert jwt.get_unverified_header(headers["authorization"].split()[1])["kid"] == "KEYID12345"
    assert body["aps"]["alert"]["title"] == "Your cut is ready" and body["project_id"] == "p1"


def test_single_failed_texts_in_her_language(r, apns):
    push.register(user_id="u", token=TOKEN_A, language="es", client=r)
    apple = Apple()
    push.cut_finished(user_id="u", project_id="p1", sources=_src(), ok=False, error="no_speech", client=r, post=apple)
    assert apple.calls[0][2]["aps"]["alert"]["body"] == "No pudimos escuchar a nadie hablando en este video"


def test_bulk_sends_once_when_all_are_done(r, apns):
    push.register(user_id="u", token=TOKEN_A, client=r)
    push.register(user_id="u", token=TOKEN_B, language="es", client=r)
    apple = Apple()
    for n in (1, 2):
        out = push.cut_finished(user_id="u", project_id=f"p{n}", sources=_src("b1", 3), ok=True, client=r, post=apple)
        assert out["reason"] == "bulk_waiting"
    # A retried job of the same video does not count twice.
    assert push.cut_finished(user_id="u", project_id="p2", sources=_src("b1", 3), ok=True, client=r,
                             post=apple)["reason"] == "bulk_waiting"
    out = push.cut_finished(user_id="u", project_id="p3", sources=_src("b1", 3), ok=True, client=r, post=apple)
    assert out["sent"] == 2
    titles = sorted(c[2]["aps"]["alert"]["title"] for c in apple.calls)
    assert titles == ["Tus 3 videos están listos ✨", "Your 3 videos are ready ✨"]
    assert all(c[1]["apns-collapse-id"] == "bulk-b1" for c in apple.calls)
    again = push.cut_finished(user_id="u", project_id="p3", sources=_src("b1", 3), ok=True, client=r, post=apple)
    assert again["reason"] == "bulk_already_told" and len(apple.calls) == 2


def test_bulk_with_a_failure_says_so(r, apns):
    push.register(user_id="u", token=TOKEN_A, client=r)
    apple = Apple()
    for n, ok in ((1, True), (2, False), (3, True)):
        push.cut_finished(user_id="u", project_id=f"p{n}", sources=_src("b2", 3), ok=ok, client=r, post=apple)
    assert apple.calls[0][2]["aps"]["alert"]["title"] == "2 ready, 1 needs attention"


@pytest.mark.parametrize("ready,failed,lang,title", [
    (9, 1, "en", "9 ready, 1 needs attention"), (1, 2, "en", "1 ready, 2 need attention"),
    (0, 3, "en", "3 videos need attention"), (1, 0, "en", "Your video is ready ✨"),
    (9, 1, "es", "9 listos, 1 necesita atención"), (1, 1, "es", "1 listo, 1 necesita atención"),
    (10, 0, "es", "Tus 10 videos están listos ✨"),
])
def test_group_titles(ready, failed, lang, title):
    assert push.group_title(ready, failed, lang) == title


def test_removed_app_token_is_forgotten(r, apns):
    push.register(user_id="u", token=TOKEN_A, client=r)
    out = push.cut_finished(user_id="u", project_id="p1", sources=_src(), ok=True, client=r,
                            post=Apple(410, '{"reason":"Unregistered"}'))
    assert out["removed"] == 1 and push._tokens("u", client=r) == {}


def test_apple_down_never_raises(r, apns):
    push.register(user_id="u", token=TOKEN_A, client=r)

    def down(*_):
        raise ConnectionError("no route")

    out = push.cut_finished(user_id="u", project_id="p1", sources=_src(), ok=True, client=r, post=down)
    assert out["sent"] == 0 and out["errors"] == ["ConnectionError"]


def test_mix_one_push(r, apns):
    push.register(user_id="u", token=TOKEN_A, client=r)
    apple = Apple()
    push.mix_finished(user_id="u", mix_id="m1", videos=6, project_id="p1", client=r, post=apple)
    assert apple.calls[0][2]["aps"]["alert"]["title"] == "Your 6 videos are ready ✨"
    assert apple.calls[0][2]["mix_id"] == "m1"


def test_configured_flag_for_the_app(r, apns):
    assert push.active(client=r) is False
    push.mark_configured(client=r)
    assert push.active(client=r) is True


def test_register_route(monkeypatch, r):
    from fastapi.testclient import TestClient
    import cutsell_app.main as api
    monkeypatch.setattr(push, "_redis", lambda client=None: r)
    client = TestClient(api.app)
    ok = client.post("/v1/push/register", json={"user_id": "u", "token": TOKEN_A, "language": "es"})
    assert ok.status_code == 200 and ok.json() == {"ok": True, "push_active": False}
    assert client.post("/v1/push/register", json={"user_id": "u", "token": "z" * 40}).status_code == 422
    assert client.get("/v1/capabilities").json()["push"] is False


def test_worker_hook_never_raises(monkeypatch):
    from cutsell_worker import worker_job

    def boom(**_):
        raise RuntimeError("redis down")

    monkeypatch.setattr(push, "cut_finished", boom)
    out = worker_job._safe_push(SimpleNamespace(user_id="u", project_id="p", sources=[]), ok=True)
    assert out["sent"] == 0
