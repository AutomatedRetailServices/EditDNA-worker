import pytest

from cutsell_worker.apple_auth import verify_apple_identity_token
from cutsell_worker.auth import create_session, resolve_session, stable_apple_user_id


class FakeRedis:
    def __init__(self):
        self.data = {}

    def setex(self, key, ttl, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)


def test_apple_subject_maps_to_stable_cutsell_user_id():
    first = stable_apple_user_id("000123.abc")
    second = stable_apple_user_id("000123.abc")
    other = stable_apple_user_id("000456.def")
    assert first == second
    assert first.startswith("usr_apple_")
    assert first != other


def test_session_can_reuse_persistent_user_identity():
    redis = FakeRedis()
    user_id = stable_apple_user_id("subject-1")
    session = create_session(user_id=user_id, client=redis)
    resolved = resolve_session(session["access_token"], client=redis)
    assert session["user_id"] == user_id
    assert resolved["user_id"] == user_id


def test_apple_verifier_refuses_to_run_without_client_id(monkeypatch):
    monkeypatch.delenv("CUTSELL_APPLE_CLIENT_ID", raising=False)
    with pytest.raises(RuntimeError, match="CUTSELL_APPLE_CLIENT_ID"):
        verify_apple_identity_token("not-a-real-token")
