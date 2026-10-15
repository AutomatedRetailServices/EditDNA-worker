"""Editor v5 sound library: the official v1 list, its signed manifest and the API route."""
from __future__ import annotations

import io
import json

import pytest

from cutsell_worker import sfx_library


class FakeS3:
    def __init__(self, manifest: dict | None = None):
        self.manifest = manifest
        self.signed: list[tuple[str, dict, int]] = []
        self.reads: list[tuple[str, str]] = []

    def get_object(self, Bucket, Key):
        self.reads.append((Bucket, Key))
        if self.manifest is None:
            raise KeyError("NoSuchKey")
        return {"Body": io.BytesIO(json.dumps(self.manifest).encode())}

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.signed.append((operation, dict(Params), ExpiresIn))
        return f"https://signed.example/{Params['Bucket']}/{Params['Key']}?e={ExpiresIn}"


@pytest.fixture
def bucket(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "cutsell-media-test")
    return "cutsell-media-test"


def test_official_v1_list():
    assert [(s.sfx_id, s.name, s.file) for s in sfx_library.SFX_LIBRARY] == [
        ("tap", "Tap", "tap_select.m4a"),
        ("good", "Good", "good.m4a"),
        ("wrong", "Wrong", "wrong.m4a"),
        ("glass", "Glass", "glass.m4a"),
        ("typing", "Typing", "typing_3s.m4a"),
        ("typing_long", "Typing (long)", "typing.m4a"),
    ]
    assert all(s.duration_sec is None for s in sfx_library.SFX_LIBRARY)
    assert all(s.license == "CC0 1.0 (Kenney Interface Sounds; Keyboard Soundpack by unicaegames)"
               for s in sfx_library.SFX_LIBRARY)
    assert sfx_library.sfx_object_key("typing") == "assets/sfx/v1/typing_3s.m4a"
    with pytest.raises(KeyError):
        sfx_library.sfx_object_key("../secrets")


def test_signed_manifest_without_stored_manifest(bucket):
    s3 = FakeS3()
    out = sfx_library.signed_manifest(client=s3)
    assert out["version"] == "v1" and len(out["sounds"]) == 6
    first = out["sounds"][0]
    assert set(first) == {"id", "name", "file", "duration_sec", "license", "url", "expires_in"}
    assert first["id"] == "tap" and first["duration_sec"] is None and first["expires_in"] == 3600
    assert first["url"].startswith("https://signed.example/cutsell-media-test/assets/sfx/v1/tap_select.m4a")
    assert s3.reads == [(bucket, "assets/sfx/v1/manifest.json")]
    assert all(op == "get_object" and p["Bucket"] == bucket and p["Key"].startswith("assets/sfx/v1/")
               for op, p, _ in s3.signed)


def test_signed_manifest_reads_durations_from_the_stored_manifest(bucket):
    s3 = FakeS3({"version": "v1", "sounds": [
        {"id": "tap", "duration_sec": 0.21}, {"id": "typing_long", "duration_sec": 9.5},
        {"id": "unknown", "duration_sec": 1.0}, {"id": "good", "duration_sec": -3}]})
    out = sfx_library.signed_manifest(expires_in=600, client=s3)
    by_id = {row["id"]: row for row in out["sounds"]}
    assert by_id["tap"]["duration_sec"] == 0.21 and by_id["typing_long"]["duration_sec"] == 9.5
    assert by_id["good"]["duration_sec"] is None and "unknown" not in by_id
    assert all(row["expires_in"] == 600 for row in out["sounds"])


def test_signed_manifest_requires_s3(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setattr(sfx_library, "load_runtime_config",
                        lambda: type("C", (), {"s3_bucket": None, "aws_region": None})())
    with pytest.raises(RuntimeError):
        sfx_library.signed_manifest(client=FakeS3())


def test_sfx_manifest_route(monkeypatch):
    from fastapi.testclient import TestClient
    import cutsell_app.main as main

    monkeypatch.setattr(main, "signed_sfx_manifest", lambda: {"version": "v1", "sounds": []})
    client = TestClient(main.app)
    assert client.get("/v1/sfx/manifest").json() == {"version": "v1", "sounds": []}

    def unavailable():
        raise RuntimeError("S3_BUCKET is required for the sound library")
    monkeypatch.setattr(main, "signed_sfx_manifest", unavailable)
    response = client.get("/v1/sfx/manifest")
    assert response.status_code == 503 and "S3_BUCKET" in response.json()["detail"]


def test_sfx_manifest_route_requires_a_session(monkeypatch):
    from fastapi.testclient import TestClient
    import cutsell_app.main as main

    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")
    monkeypatch.setattr(main, "signed_sfx_manifest", lambda: {"version": "v1", "sounds": []})
    response = TestClient(main.app).get("/v1/sfx/manifest")
    assert response.status_code == 401
