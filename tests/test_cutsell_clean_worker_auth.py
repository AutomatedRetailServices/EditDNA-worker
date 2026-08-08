from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import cutsell_app.auth_middleware as middleware
from cutsell_app.auth_middleware import AuthScopeMiddleware
from cutsell_worker.auth import create_session, resolve_session


class FakeRedis:
    def __init__(self):
        self.data = {}
    def setex(self, key, ttl, value):
        self.data[key] = value
        return True
    def get(self, key):
        return self.data.get(key)


def test_opaque_session_stores_only_hashed_lookup_and_resolves_user():
    redis = FakeRedis()
    session = create_session(client=redis)
    assert session["user_id"].startswith("usr_")
    assert len(session["access_token"]) > 40
    assert session["access_token"] not in " ".join(redis.data.keys())
    resolved = resolve_session(session["access_token"], client=redis)
    assert resolved["user_id"] == session["user_id"]


def _secure_app(monkeypatch):
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")
    monkeypatch.setattr(
        middleware,
        "resolve_session",
        lambda token: {"user_id": "usr_owner"} if token == "good-token" else (_ for _ in ()).throw(PermissionError("invalid token")),
    )
    def fake_job(job_id, *, user_id=None, **_kwargs):
        if job_id == "foreign":
            raise PermissionError("job does not belong to this user")
        return object()
    monkeypatch.setattr(middleware, "fetch_job_snapshot", fake_job)

    app = FastAPI()
    app.add_middleware(AuthScopeMiddleware)

    @app.post("/private")
    async def private(request: Request, payload: dict):
        return {"auth_user_id": request.state.auth_user_id, "claimed": payload.get("user_id")}

    @app.get("/query")
    async def query(request: Request, user_id: str):
        return {"auth_user_id": request.state.auth_user_id, "claimed": user_id}

    @app.get("/v1/jobs/{job_id}")
    async def job_route(job_id: str):
        return {"job_id": job_id}

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_route(job_id: str):
        return {"job_id": job_id, "state": "canceled"}

    @app.get("/v1/healthz")
    async def health():
        return {"ok": True}

    return TestClient(app)


def test_auth_middleware_requires_token_and_rejects_cross_user_body(monkeypatch):
    client = _secure_app(monkeypatch)
    missing = client.post("/private", json={"user_id": "usr_owner"})
    assert missing.status_code == 401

    mismatch = client.post(
        "/private",
        json={"user_id": "usr_other"},
        headers={"Authorization": "Bearer good-token"},
    )
    assert mismatch.status_code == 403

    allowed = client.post(
        "/private",
        json={"user_id": "usr_owner"},
        headers={"Authorization": "Bearer good-token"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["auth_user_id"] == "usr_owner"


def test_auth_middleware_checks_query_scope_and_keeps_health_public(monkeypatch):
    client = _secure_app(monkeypatch)
    assert client.get("/v1/healthz").status_code == 200
    bad = client.get("/query?user_id=usr_other", headers={"Authorization": "Bearer good-token"})
    assert bad.status_code == 403
    good = client.get("/query?user_id=usr_owner", headers={"Authorization": "Bearer good-token"})
    assert good.status_code == 200


def test_auth_middleware_binds_job_status_and_cancel_to_bearer_user(monkeypatch):
    client = _secure_app(monkeypatch)
    headers = {"Authorization": "Bearer good-token"}
    assert client.get("/v1/jobs/owned", headers=headers).status_code == 200
    assert client.post("/v1/jobs/owned/cancel", headers=headers).status_code == 200
    assert client.get("/v1/jobs/foreign", headers=headers).status_code == 403
    assert client.post("/v1/jobs/foreign/cancel", headers=headers).status_code == 403
