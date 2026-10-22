"""Teleprompter · Split for Mix: her words only, one call, a clear error when Haiku is not reachable."""
from fastapi.testclient import TestClient

from cutsell_app import script_routes as sr

SCRIPT = ("Stop wasting time with old straps. These Rhino retractable straps lock in one pull. "
          "I hauled my whole kayak with zero slipping. Tap the orange cart before they're gone.")


def _fake(parts):
    def call(body):
        assert body["model"] == "claude-haiku-5-5"
        assert "Rhino" in body["messages"][0]["content"]
        return parts, {"input_tokens": 600, "output_tokens": 200}, None
    return call


def test_split_keeps_her_sentences():
    out = sr.split(SCRIPT, "Rhino · Straps", call=_fake({
        "hook": "Stop wasting time with old straps.",
        "body": "These Rhino retractable straps lock in one pull.",
        "benefit": "I hauled my whole kayak with zero slipping.",
        "cta": "Tap the orange cart before they're gone."}))
    assert out["parts"]["hook"] == "Stop wasting time with old straps."
    assert out["parts"]["cta"].startswith("Tap the orange cart")
    assert 0 < out["cost_usd"] < 0.001


def test_invented_sentences_are_dropped():
    out = sr.split(SCRIPT, None, call=_fake({
        "hook": "Stop wasting time with old straps. You NEED these now!",
        "body": "These Rhino retractable straps lock in one pull.",
        "benefit": "", "cta": "Buy two today."}))
    assert out["parts"]["hook"] == "Stop wasting time with old straps."
    assert out["parts"]["cta"] == ""


def test_route_reports_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from cutsell_app.main import app
    response = TestClient(app).post("/v1/scripts/split", json={"user_id": "u1", "script": SCRIPT})
    assert response.status_code == 503 and "split_unavailable:no_key" in response.text
