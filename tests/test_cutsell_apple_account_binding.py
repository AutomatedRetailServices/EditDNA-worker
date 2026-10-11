"""Sign in with Apple keeps the guest's projects, logout signs out one phone, sessions in use
do not expire, and account deletion frees the Apple ID (2026-10-11, Swanny's point 2)."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cutsell_app.auth_routes as routes
from cutsell_worker import auth
from cutsell_worker.apple_auth import AppleIdentity


class FakeRedis:
    def __init__(self):
        self.kv, self.sets, self.ttl = {}, {}, {}

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def setex(self, key, ttl, value):
        self.kv[key] = value
        self.ttl[key] = ttl

    def delete(self, *keys):
        for key in keys:
            self.kv.pop(key, None)
            self.sets.pop(key, None)

    def expire(self, key, seconds):
        self.ttl[key] = seconds

    def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)

    def srem(self, key, *values):
        self.sets.get(key, set()).difference_update(values)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def incr(self, key):
        self.kv[key] = int(self.kv.get(key, 0)) + 1
        return self.kv[key]

    def pipeline(self):
        outer = self

        class Pipe:
            def __getattr__(self, name):
                return getattr(outer, name)

            def execute(self):
                return []
        return Pipe()


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(auth, "_redis_client", lambda client=None: client or fake)
    return fake


# ---- binding rules ----------------------------------------------------------------------

def test_first_apple_sign_in_adopts_the_guest_account(redis):
    guest = auth.create_session()["user_id"]
    user_id, outcome = auth.resolve_apple_account("apple-sub-1", guest_user_id=guest)
    assert (user_id, outcome) == (guest, "guest_upgraded")
    # Next sign-in with the same Apple ID (new phone, reinstall): same account, same projects.
    assert auth.resolve_apple_account("apple-sub-1") == (guest, "existing_account")


def test_apple_sign_in_without_guest_creates_a_stable_account(redis):
    user_id, outcome = auth.resolve_apple_account("apple-sub-2")
    assert outcome == "new_account" and user_id == auth.stable_apple_user_id("apple-sub-2")
    assert auth.resolve_apple_account("apple-sub-2") == (user_id, "existing_account")


def test_other_guest_on_existing_apple_account_is_not_merged(redis):
    first_guest = auth.create_session()["user_id"]
    auth.resolve_apple_account("apple-sub-3", guest_user_id=first_guest)
    second_guest = auth.create_session()["user_id"]
    assert auth.resolve_apple_account("apple-sub-3", guest_user_id=second_guest) == (
        first_guest, "existing_account_guest_not_merged")


def test_a_guest_bound_to_one_apple_id_cannot_be_taken_by_another(redis):
    guest = auth.create_session()["user_id"]
    auth.resolve_apple_account("apple-sub-4", guest_user_id=guest)
    user_id, outcome = auth.resolve_apple_account("apple-sub-5", guest_user_id=guest)
    assert outcome == "new_account" and user_id != guest


def test_apple_accounts_are_never_adopted_as_guests(redis):
    apple_user = auth.stable_apple_user_id("apple-sub-6")
    user_id, outcome = auth.resolve_apple_account("apple-sub-7", guest_user_id=apple_user)
    assert outcome == "new_account" and user_id != apple_user


def test_deleting_the_account_frees_the_apple_id(redis):
    guest = auth.create_session()["user_id"]
    auth.resolve_apple_account("apple-sub-8", guest_user_id=guest)
    assert auth.delete_apple_binding(user_id=guest) is True
    assert auth.resolve_apple_account("apple-sub-8")[1] == "new_account"


# ---- sessions -------------------------------------------------------------------------

def test_session_in_use_is_extended(redis):
    session = auth.create_session()
    redis.ttl[auth.session_key(session["access_token"])] = 5
    auth.resolve_session(session["access_token"])
    assert redis.ttl[auth.session_key(session["access_token"])] == auth.SESSION_TTL_SEC


def test_logout_revokes_only_this_phone(redis):
    phone_a = auth.create_session(user_id="usr_apple_x")
    phone_b = auth.create_session(user_id="usr_apple_x")
    assert auth.revoke_session(phone_a["access_token"]) == {"status": "signed_out"}
    with pytest.raises(PermissionError):
        auth.resolve_session(phone_a["access_token"])
    assert auth.resolve_session(phone_b["access_token"])["user_id"] == "usr_apple_x"
    assert auth.revoke_session(phone_a["access_token"]) == {"status": "already_signed_out"}


# ---- routes -------------------------------------------------------------------------

@pytest.fixture
def client(redis, monkeypatch):
    monkeypatch.setattr(routes, "verify_apple_identity_token",
                        lambda token, nonce=None: AppleIdentity(subject=token, email=None, email_verified=None))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_apple_route_upgrades_the_guest_sent_by_the_phone(client, redis):
    guest = auth.create_session()
    r = client.post("/v1/auth/apple", json={"identity_token": "sub-route-1"},
                    headers={"Authorization": f"Bearer {guest['access_token']}"})
    body = r.json()
    assert r.status_code == 200
    assert body["user_id"] == guest["user_id"] and body["account_outcome"] == "guest_upgraded"
    assert body["access_token"] != guest["access_token"]


def test_apple_route_ignores_a_fake_guest_token(client):
    r = client.post("/v1/auth/apple", json={"identity_token": "sub-route-2"},
                    headers={"Authorization": "Bearer not-a-real-session"})
    assert r.status_code == 200 and r.json()["account_outcome"] == "new_account"


def test_apple_route_reports_guest_projects_not_merged(client, redis, monkeypatch):
    import cutsell_worker.project_store as project_store
    first = auth.create_session()
    client.post("/v1/auth/apple", json={"identity_token": "sub-route-3"},
                headers={"Authorization": f"Bearer {first['access_token']}"})
    second = auth.create_session()
    monkeypatch.setattr(project_store, "list_projects", lambda user_id, limit=100: [{"p": 1}, {"p": 2}])
    body = client.post("/v1/auth/apple", json={"identity_token": "sub-route-3"},
                       headers={"Authorization": f"Bearer {second['access_token']}"}).json()
    assert body["account_outcome"] == "existing_account_guest_not_merged"
    assert body["user_id"] == first["user_id"] and body["guest_projects_not_merged"] == 2


def test_logout_route(client, redis):
    session = auth.create_session()
    assert client.post("/v1/auth/logout", headers={"Authorization": f"Bearer {session['access_token']}"}).json() == {
        "status": "signed_out"}
    assert client.post("/v1/auth/logout").status_code == 401
