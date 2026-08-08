from types import SimpleNamespace

import pytest

import cutsell_worker.uploads as uploads
from cutsell_app import main


class FakeS3:
    def __init__(self):
        self.calls = []

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        self.calls.append((operation, Params, ExpiresIn))
        return "https://signed.example/source.mp4?token=fresh"


def _config():
    return SimpleNamespace(s3_bucket="bucket", aws_region="us-east-1")


def _owned_uri(user_id="user", project_id="project"):
    prefix = uploads.scoped_upload_prefix(user_id=user_id, project_id=project_id)
    return f"s3://bucket/{prefix}abc-video.mp4"


def test_source_playback_signing_requires_exact_user_project_scope(monkeypatch):
    monkeypatch.setattr(uploads, "load_runtime_config", _config)
    client = FakeS3()
    signed = uploads.create_presigned_source_download(
        _owned_uri(),
        user_id="user",
        project_id="project",
        expires_in=900,
        client=client,
    )
    assert signed == {"url": "https://signed.example/source.mp4?token=fresh", "expires_in": 900}
    assert client.calls == [(
        "get_object",
        {"Bucket": "bucket", "Key": _owned_uri().split("s3://bucket/", 1)[1]},
        900,
    )]

    with pytest.raises(ValueError, match="outside allowed CutSell upload scope"):
        uploads.create_presigned_source_download(
            _owned_uri(),
            user_id="another-user",
            project_id="project",
            client=client,
        )


def test_source_playback_signing_rejects_invalid_expiry(monkeypatch):
    monkeypatch.setattr(uploads, "load_runtime_config", _config)
    with pytest.raises(ValueError, match="expiry"):
        uploads.create_presigned_source_download(
            _owned_uri(), user_id="user", project_id="project", expires_in=10, client=FakeS3()
        )


def test_draft_hydration_adds_fresh_playback_url_without_mutating_canonical_source(monkeypatch):
    uri = _owned_uri()
    snapshot = {
        "project_id": "project",
        "user_id": "user",
        "revision": 3,
        "draft": {"selected": []},
        "sources": [{
            "source_asset_id": "src",
            "uri": uri,
            "timeline_assets": {"status": "ready"},
        }],
    }
    monkeypatch.setattr(main, "sign_timeline_assets", lambda assets: {**assets, "signed_url_status": "ready"})
    monkeypatch.setattr(
        main,
        "create_presigned_source_download",
        lambda *args, **kwargs: {"url": "https://fresh.example/video", "expires_in": 1800},
    )

    hydrated = main._hydrate_snapshot_assets(snapshot)
    source = hydrated["sources"][0]
    assert source["uri"] == uri
    assert source["playback_url"] == "https://fresh.example/video"
    assert source["playback_expires_in"] == 1800
    assert source["playback_url_status"] == "ready"
    assert "playback_url" not in snapshot["sources"][0]


def test_draft_hydration_fails_open_when_playback_signing_is_unavailable(monkeypatch):
    snapshot = {
        "project_id": "project",
        "user_id": "user",
        "revision": 1,
        "draft": {"selected": []},
        "sources": [{"source_asset_id": "src", "uri": _owned_uri()}],
    }

    def fail(*args, **kwargs):
        raise RuntimeError("s3 temporarily unavailable")

    monkeypatch.setattr(main, "create_presigned_source_download", fail)
    hydrated = main._hydrate_snapshot_assets(snapshot)
    source = hydrated["sources"][0]
    assert source["uri"] == snapshot["sources"][0]["uri"]
    assert source["playback_url"] is None
    assert source["playback_url_status"] == "degraded"
    assert source["playback_url_reason"] == "RuntimeError"
