"""You · Account card: GET/PUT /v1/account/me (name kept per account; email from the account store)."""
import fakeredis
from fastapi.testclient import TestClient


def test_name_round_trip(monkeypatch):
    import cutsell_app.account_me_routes as routes
    import cutsell_app.main as api
    r = fakeredis.FakeRedis()
    monkeypatch.setattr(routes, "_redis", lambda: r)
    monkeypatch.setattr(routes, "_email", lambda user_id: "swanny@example.com" if user_id == "u1" else None)
    monkeypatch.setenv("CUTSELL_AUTH_DISABLED", "1")
    client = TestClient(api.app)
    assert client.get("/v1/account/me", params={"user_id": "u1"}).json() == {"user_id": "u1", "email": "swanny@example.com", "name": None}
    got = client.put("/v1/account/me", json={"user_id": "u1", "name": "  Swanny   S "}).json()
    assert got["name"] == "Swanny S" and got["email"] == "swanny@example.com"
    assert client.get("/v1/account/me", params={"user_id": "u2"}).json()["name"] is None   # per account
    assert client.put("/v1/account/me", json={"user_id": "u1", "name": ""}).status_code == 422
