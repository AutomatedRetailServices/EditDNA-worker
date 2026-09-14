from fastapi.testclient import TestClient

import cutsell_app.main as api
import cutsell_app.notification_routes as routes
import cutsell_worker.notifications as notifications


class FakeRedis:
    def __init__(self):
        self.data = {}
    def get(self, key):
        return self.data.get(key)
    def set(self, key, value):
        self.data[key] = value
        return True


def test_notification_outbox_is_user_scoped_and_bounded():
    redis = FakeRedis()
    for index in range(105):
        notifications.publish_notification(
            user_id="u1",
            project_id=f"p{index}",
            kind="draft_ready",
            payload={"index": index},
            client=redis,
        )
    items = notifications.list_notifications(user_id="u1", limit=100, client=redis)
    assert len(items) == 100
    assert items[0]["payload"]["index"] == 104
    assert notifications.list_notifications(user_id="other", client=redis) == []


def test_notification_api_returns_user_events(monkeypatch):
    monkeypatch.setattr(
        routes,
        "list_notifications",
        lambda **kwargs: [{"notification_id": "ntf_1", "kind": "render_finished"}],
    )
    response = TestClient(api.app).get("/v1/notifications?user_id=u1")
    assert response.status_code == 200
    assert response.json()["notifications"][0]["kind"] == "render_finished"
