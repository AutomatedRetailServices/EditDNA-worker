"""Account deletion is complete (PO 6 Nov 2026): every project (no 100 cap), the catalog, mixes, plan,
render lists, clean-audio states and the account name go too; if any project fails, nothing is reported as
deleted and the session stays alive so the app can ask again."""
from types import SimpleNamespace

import fakeredis
import pytest

import cutsell_worker.account_lifecycle as lifecycle
from cutsell_worker import mix_store, post_plan, product_catalog
from cutsell_worker.project_store import project_index_key, project_key
from cutsell_worker.render_versions import render_versions_key


@pytest.fixture
def r(monkeypatch):
    client = fakeredis.FakeRedis()
    monkeypatch.setattr(lifecycle, "load_runtime_config",
                        lambda: SimpleNamespace(s3_bucket=None, aws_region="auto", redis_url="redis://x",
                                                database_url=None))
    monkeypatch.setattr(lifecycle, "revoke_all_sessions", lambda **k: {"revoked": 1})
    monkeypatch.setattr(lifecycle, "delete_apple_binding", lambda **k: False)
    for name in ("FALLBACK_S3_BUCKET", "FALLBACK_AWS_ACCESS_KEY_ID", "FALLBACK_AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    return client


def _seed(r, user, n):
    for i in range(n):
        pid = f"prj_{i}"
        r.set(project_key(user_id=user, project_id=pid), '{"project_id": "%s", "user_id": "%s"}' % (pid, user))
        r.zadd(project_index_key(user_id=user), {pid: i})
        r.set(render_versions_key(user_id=user, project_id=pid), "[]")
    r.set(product_catalog.catalog_key(user), "{}")
    r.set(post_plan.plan_key(user), "{}")
    r.zadd(mix_store.index_key(user), {"mix_1": 1})
    r.set(mix_store.mix_key(user, "mix_1"), "{}")


def test_every_project_and_every_account_record_is_deleted(r):
    _seed(r, "usr_me", 130)
    _seed(r, "usr_other", 2)
    before_other = sorted(k for k in r.keys() if b"usr_other" not in k)
    out = lifecycle.delete_account_data(user_id="usr_me", redis_client=r, s3_client=None)
    assert out["status"] == "deleted" and len(out["projects_deleted"]) == 130 and out["mixes_deleted"] == 1
    # only the other user's records are left
    left = set(r.keys())
    for i in range(2):
        assert project_key(user_id="usr_other", project_id=f"prj_{i}").encode() in left
    assert product_catalog.catalog_key("usr_other").encode() in left
    assert mix_store.mix_key("usr_other", "mix_1").encode() in left
    assert len(left) == 2 * 2 + 5 and before_other   # her 2 projects + 2 render lists + index, catalog, plan, mix index, mix


def test_a_failed_project_keeps_the_account_and_says_partial(r, monkeypatch):
    _seed(r, "usr_me", 3)
    real = lifecycle.delete_project_data
    revoked = []
    monkeypatch.setattr(lifecycle, "revoke_all_sessions", lambda **k: revoked.append(1))

    def flaky(**kwargs):
        if kwargs["project_id"] == "prj_1":
            raise ConnectionError("storage down")
        return real(**kwargs)

    monkeypatch.setattr(lifecycle, "delete_project_data", flaky)
    out = lifecycle.delete_account_data(user_id="usr_me", redis_client=r, s3_client=None)
    assert out["status"] == "partial" and out["projects_failed"] == [{"project_id": "prj_1", "error": "ConnectionError"}]
    assert not revoked and r.exists(product_catalog.catalog_key("usr_me"))
    monkeypatch.setattr(lifecycle, "delete_project_data", real)
    again = lifecycle.delete_account_data(user_id="usr_me", redis_client=r, s3_client=None)
    assert again["status"] == "deleted" and not r.exists(product_catalog.catalog_key("usr_me"))


def test_the_route_answers_503_when_deletion_is_partial(monkeypatch):
    from fastapi import HTTPException
    import cutsell_app.auth_routes as routes
    monkeypatch.setattr(routes, "delete_account_data", lambda **k: {"status": "partial"})
    with pytest.raises(HTTPException) as err:
        routes.delete_authenticated_account(SimpleNamespace(user_id="u", confirmation="DELETE MY ACCOUNT"))
    assert err.value.status_code == 503


def test_apple_sign_in_is_revoked_at_deletion_when_the_key_is_configured(monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cutsell_worker import apple_auth, auth

    r = fakeredis.FakeRedis()
    for name in ("CUTSELL_APPLE_TEAM_ID", "CUTSELL_APPLE_KEY_ID", "CUTSELL_APPLE_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUTSELL_APPLE_CLIENT_ID", "ai.cutsell.app")
    # not configured: nothing is called, sign-in unchanged
    assert apple_auth.exchange_authorization_code("code", post=lambda *a, **k: 1 / 0) is None
    assert auth.revoke_apple_token(user_id="u", client=r) == "not_configured"

    pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    monkeypatch.setenv("CUTSELL_APPLE_TEAM_ID", "TEAM")
    monkeypatch.setenv("CUTSELL_APPLE_KEY_ID", "KEY")
    monkeypatch.setenv("CUTSELL_APPLE_PRIVATE_KEY", pem)
    calls = []

    def post(url, data=None, timeout=None):
        calls.append((url, data))
        return SimpleNamespace(status_code=200, json=lambda: {"refresh_token": "rt_1"})

    token = apple_auth.exchange_authorization_code("code_1", post=post)
    assert token == "rt_1" and calls[0][0] == apple_auth.APPLE_TOKEN_URL and calls[0][1]["code"] == "code_1"
    auth.store_apple_refresh_token(user_id="u", token=token, client=r)
    assert auth.revoke_apple_token(user_id="u", client=r, post=post) == "revoked"
    assert calls[1][0] == apple_auth.APPLE_REVOKE_URL and calls[1][1]["token"] == "rt_1"
    assert not r.exists(auth.apple_refresh_key("u"))


def test_every_file_in_her_drawers_is_deleted_even_without_a_project_record(r, monkeypatch):
    from cutsell_worker.account_lifecycle import _user_prefixes

    class S3:
        def __init__(self, keys):
            self.keys = list(keys)

        def list_objects_v2(self, **kw):
            return {"Contents": [{"Key": k} for k in self.keys if k.startswith(kw["Prefix"])], "IsTruncated": False}

        def delete_objects(self, **kw):
            for item in kw["Delete"]["Objects"]:
                self.keys.remove(item["Key"])

    mine = [p + "orphan_project/video.mp4" for p in _user_prefixes(user_id="usr_me")]
    other = [p + "x/video.mp4" for p in _user_prefixes(user_id="usr_other")]
    s3 = S3(mine + other)
    monkeypatch.setattr(lifecycle, "load_runtime_config",
                        lambda: SimpleNamespace(s3_bucket="cutsell-videos", aws_region="auto", redis_url="redis://x",
                                                database_url=None))
    out = lifecycle.delete_account_data(user_id="usr_me", redis_client=r, s3_client=s3)
    assert out["status"] == "deleted" and sorted(s3.keys) == sorted(other)
    assert any(p.startswith("cutsell/pending-review/") for p in _user_prefixes(user_id="usr_me"))
