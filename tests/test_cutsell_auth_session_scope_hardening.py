"""A session can only ever act as its own user, however the request is shaped, and anonymous
accounts are rate-limited (2026-10-11 security review findings HIGH #1-3 and MEDIUM #4)."""
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

import cutsell_app.auth_middleware as middleware
import cutsell_app.auth_routes as auth_routes
from cutsell_worker import auth


class Body(BaseModel):
    user_id: str


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")

    def resolve(token):
        if token == "tokA":
            return {"user_id": "usr_A"}
        raise PermissionError("invalid or expired bearer token")

    monkeypatch.setattr(middleware, "resolve_session", resolve)
    app = FastAPI()

    @app.get("/v1/q")
    def query_route(user_id: str):
        return {"route_user": user_id}

    @app.post("/v1/b")
    def body_route(payload: Body, request: Request):
        return {"route_user": payload.user_id, "session_user": request.state.auth_user_id}

    app.add_middleware(middleware.AuthScopeMiddleware)
    return TestClient(app)


AUTH = ("Authorization", "Bearer tokA")


def test_own_user_still_works(client):
    assert client.get("/v1/q?user_id=usr_A", headers=[AUTH]).json() == {"route_user": "usr_A"}
    r = client.post("/v1/b", content=b'{"user_id":"usr_A"}', headers=[AUTH, ("Content-Type", "application/json")])
    assert r.status_code == 200 and r.json()["route_user"] == "usr_A"


def test_other_user_in_query_is_rejected(client):
    assert client.get("/v1/q?user_id=usr_B", headers=[AUTH]).status_code == 403


@pytest.mark.parametrize("query", ["user_id=usr_A&user_id=usr_B", "user_id=usr_B&user_id=usr_A", "user%5Fid=usr_B"])
def test_duplicate_or_encoded_query_user_is_rejected(client, query):
    # FastAPI binds the last duplicate; every value must be the session's user.
    assert client.get(f"/v1/q?{query}", headers=[AUTH]).status_code == 403


@pytest.mark.parametrize("content_type", ["application/json", "application/vnd.x+json", "application/json; charset=utf-8",
                                          "text/plain", "APPLICATION/JSON"])
def test_other_user_in_body_is_rejected_whatever_the_content_type(client, content_type):
    r = client.post("/v1/b", content=b'{"user_id":"usr_B"}', headers=[AUTH, ("Content-Type", content_type)])
    assert r.status_code == 403


def test_duplicated_content_type_header_cannot_hide_the_body(client):
    r = client.post("/v1/b", content=b'{"user_id":"usr_B"}',
                    headers=[AUTH, ("Content-Type", "application/json"), ("Content-Type", "text/plain")])
    assert r.status_code == 403


def test_escaped_or_duplicated_json_key_is_rejected(client):
    for raw in (b'{"user\\u005fid":"usr_B"}', b'{"user_id":"usr_A","user_id":"usr_B"}'):
        r = client.post("/v1/b", content=raw, headers=[AUTH, ("Content-Type", "application/json")])
        assert r.status_code == 403


def test_null_user_in_body_is_rejected(client):
    r = client.post("/v1/b", content=b'{"user_id":null}', headers=[AUTH, ("Content-Type", "application/json")])
    assert r.status_code == 403


def test_no_token_is_rejected(client):
    assert client.get("/v1/q?user_id=usr_A").status_code == 401


# ---- anonymous account rate limit -------------------------------------------------------

class FakeRedis:
    def __init__(self):
        self.data, self.ttl = {}, {}

    def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]

    def expire(self, key, seconds):
        self.ttl[key] = seconds

    def setex(self, key, ttl, value):
        self.data[key] = value


NOW = datetime(2026, 10, 11, 15, 0, tzinfo=timezone.utc)


def test_per_network_hourly_limit(monkeypatch):
    for name in ("CUTSELL_ANON_SESSIONS_PER_IP_PER_HOUR", "CUTSELL_ANON_SESSIONS_PER_IP_PER_DAY",
                 "CUTSELL_ANON_SESSIONS_GLOBAL_PER_HOUR"):
        monkeypatch.delenv(name, raising=False)
    redis = FakeRedis()
    results = [auth.check_anonymous_session_allowance(client_ip="203.0.113.7", client=redis, now=NOW)
               for _ in range(auth.ANON_PER_IP_HOUR_DEFAULT + 1)]
    assert all(ok for ok, _ in results[:-1])
    assert results[-1] == (False, "anonymous_signup_network_limit")
    # Another network is unaffected.
    assert auth.check_anonymous_session_allowance(client_ip="198.51.100.9", client=redis, now=NOW)[0]
    # The IP itself is never stored, only a hash.
    assert not any("203.0.113.7" in key for key in redis.data)


def test_global_limit_applies_without_a_trusted_ip(monkeypatch):
    monkeypatch.setenv("CUTSELL_ANON_SESSIONS_GLOBAL_PER_HOUR", "3")
    redis = FakeRedis()
    got = [auth.check_anonymous_session_allowance(client_ip=None, client=redis, now=NOW)[0] for _ in range(4)]
    assert got == [True, True, True, False]


def test_zero_disables_a_limit(monkeypatch):
    monkeypatch.setenv("CUTSELL_ANON_SESSIONS_GLOBAL_PER_HOUR", "0")
    monkeypatch.setenv("CUTSELL_ANON_SESSIONS_PER_IP_PER_HOUR", "0")
    monkeypatch.setenv("CUTSELL_ANON_SESSIONS_PER_IP_PER_DAY", "0")
    redis = FakeRedis()
    assert all(auth.check_anonymous_session_allowance(client_ip="203.0.113.7", client=redis, now=NOW)[0]
               for _ in range(50))


def test_session_route_returns_429_when_limited(monkeypatch):
    monkeypatch.setattr(auth_routes, "check_anonymous_session_allowance",
                        lambda client_ip: (False, "anonymous_signup_network_limit"))
    monkeypatch.setattr(auth_routes, "create_session", lambda: pytest.fail("must not create a session"))
    app = FastAPI()
    app.include_router(auth_routes.router)
    r = TestClient(app).post("/v1/auth/session", headers={"CF-Connecting-IP": "203.0.113.7"})
    assert r.status_code == 429
    assert "try again later" in r.json()["detail"]


def test_session_route_uses_only_trusted_ip_headers(monkeypatch):
    seen = []
    monkeypatch.setattr(auth_routes, "check_anonymous_session_allowance",
                        lambda client_ip: (seen.append(client_ip) or (True, "allowed")))
    monkeypatch.setattr(auth_routes, "create_session", lambda: {"user_id": "usr_x", "access_token": "t"})
    app = FastAPI()
    app.include_router(auth_routes.router)
    c = TestClient(app)
    c.post("/v1/auth/session", headers={"X-Forwarded-For": "1.2.3.4"})          # forgeable: ignored
    c.post("/v1/auth/session", headers={"CF-Connecting-IP": "203.0.113.7"})
    c.post("/v1/auth/session", headers={"True-Client-IP": "198.51.100.9"})
    assert seen == [None, "203.0.113.7", "198.51.100.9"]
