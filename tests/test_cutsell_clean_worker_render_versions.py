import json

from fastapi.testclient import TestClient

import cutsell_app.main as api
import cutsell_app.render_version_routes as routes
import cutsell_worker.render_versions as versions


class FakeRedis:
    def __init__(self):
        self.data = {}
    def get(self, key):
        return self.data.get(key)
    def set(self, key, value):
        self.data[key] = value
        return True


class FakeS3:
    def __init__(self):
        self.calls = []
    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.calls.append((operation, Params, ExpiresIn))
        return "https://download.invalid/version.mp4"


def test_render_versions_are_scoped_and_bounded(monkeypatch):
    redis = FakeRedis()
    for index in range(25):
        versions.add_render_version(
            user_id="user-1",
            project_id="project-1",
            export_uri=f"s3://bucket/cutsell/exports/u/p/{index}.mp4",
            size_bytes=index,
            selected_count=2,
            client=redis,
        )
    history = versions.list_render_versions(user_id="user-1", project_id="project-1", client=redis)
    assert len(history) == versions.MAX_RENDER_VERSIONS == 20
    assert history[0]["export_uri"].endswith("24.mp4")
    assert versions.list_render_versions(user_id="other", project_id="project-1", client=redis) == []


def test_sign_render_version_allows_only_cutsell_export_prefix(monkeypatch):
    monkeypatch.setattr(
        versions,
        "load_runtime_config",
        lambda: type("Cfg", (), {"s3_bucket": "bucket", "aws_region": "us-east-1"})(),
    )
    s3 = FakeS3()
    url = versions.sign_export_uri(
        "s3://bucket/cutsell/exports/u/p/file.mp4",
        client=s3,
    )
    assert url == "https://download.invalid/version.mp4"
    assert s3.calls[0][1]["Key"].startswith("cutsell/exports/")
    try:
        versions.sign_export_uri("s3://bucket/cutsell/uploads/u/p/file.mp4", client=s3)
    except ValueError as exc:
        assert "outside CutSell exports" in str(exc)
    else:
        raise AssertionError("non-export S3 key must be rejected")


def test_render_version_api_returns_fresh_urls(monkeypatch):
    monkeypatch.setattr(
        routes,
        "hydrated_render_versions",
        lambda **kwargs: [{
            "render_version_id": "rv_1",
            "project_id": kwargs["project_id"],
            "user_id": kwargs["user_id"],
            "export_uri": "s3://bucket/cutsell/exports/file.mp4",
            "download_url": "https://download.invalid/fresh.mp4",
            "download_url_status": "ready",
        }],
    )
    response = TestClient(api.app).get("/v1/projects/p1/renders?user_id=u1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["project_id"] == "p1"
    assert payload["renders"][0]["download_url_status"] == "ready"
