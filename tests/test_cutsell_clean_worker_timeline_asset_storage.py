from pathlib import Path
from types import SimpleNamespace

import cutsell_app.main as api
import cutsell_worker.timeline_asset_storage as storage


class FakeS3:
    def __init__(self):
        self.uploads = []
        self.objects = []
        self.signs = []

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.uploads.append((path, bucket, key, ExtraArgs or {}))

    def put_object(self, **kwargs):
        self.objects.append(kwargs)

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.signs.append((operation, Params, ExpiresIn))
        return f"https://signed.invalid/{Params['Key']}?ttl={ExpiresIn}"


def _config():
    return SimpleNamespace(s3_bucket="bucket", aws_region="us-east-1")


def test_store_timeline_assets_uses_scoped_prefix_and_content_types(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "load_runtime_config", _config)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    client = FakeS3()

    result = storage.store_timeline_assets(
        user_id="user-1",
        project_id="project-1",
        source_asset_id="src-1",
        filmstrip=({"time": 0.0, "path": str(frame)},),
        waveform=(0.1, 0.8, 0.2),
        client=client,
    )

    assert result["status"] == "ready"
    assert result["filmstrip"][0]["uri"].startswith("s3://bucket/cutsell/timeline-assets/")
    assert result["waveform_uri"].startswith("s3://bucket/cutsell/timeline-assets/")
    assert client.uploads[0][3]["ContentType"] == "image/jpeg"
    assert client.objects[0]["ContentType"] == "application/json"
    assert b'"peaks"' in client.objects[0]["Body"]


def test_sign_timeline_assets_generates_fresh_urls_and_rejects_foreign_scope(monkeypatch):
    monkeypatch.setattr(storage, "load_runtime_config", _config)
    client = FakeS3()
    metadata = {
        "status": "ready",
        "filmstrip": [{
            "time": 1.5,
            "uri": "s3://bucket/cutsell/timeline-assets/u/p/s/frame-000.jpg",
        }],
        "waveform_uri": "s3://bucket/cutsell/timeline-assets/u/p/s/waveform.json",
        "waveform_bucket_count": 256,
    }
    signed = storage.sign_timeline_assets(metadata, client=client, expires_in=600)
    assert signed["filmstrip"][0]["download_url"].startswith("https://signed.invalid/")
    assert signed["waveform_download_url"].startswith("https://signed.invalid/")
    assert signed["signed_url_expires_in"] == 600

    foreign = dict(metadata)
    foreign["waveform_uri"] = "s3://other/cutsell/timeline-assets/u/p/s/waveform.json"
    try:
        storage.sign_timeline_assets(foreign, client=client)
    except ValueError as exc:
        assert "outside allowed scope" in str(exc)
    else:
        raise AssertionError("foreign timeline asset bucket must be rejected")


def test_recovery_hydrates_asset_urls_but_fails_open(monkeypatch):
    snapshot = {
        "project_id": "project-1",
        "user_id": "user-1",
        "revision": 2,
        "draft": {"project_id": "project-1", "selected": []},
        "sources": [{
            "source_asset_id": "src-1",
            "timeline_assets": {"status": "ready", "filmstrip": [], "waveform_uri": "s3://bucket/x"},
        }],
    }
    monkeypatch.setattr(
        api,
        "sign_timeline_assets",
        lambda metadata: {**metadata, "waveform_download_url": "https://fresh.invalid/waveform"},
    )
    hydrated = api._hydrate_snapshot_assets(snapshot)
    assert hydrated["sources"][0]["timeline_assets"]["waveform_download_url"].startswith("https://fresh.invalid")
    assert "waveform_download_url" not in snapshot["sources"][0]["timeline_assets"]

    def fail(_metadata):
        raise RuntimeError("temporary signing outage")

    monkeypatch.setattr(api, "sign_timeline_assets", fail)
    degraded = api._hydrate_snapshot_assets(snapshot)
    assert degraded["sources"][0]["timeline_assets"]["status"] == "ready"
    assert degraded["sources"][0]["timeline_assets"]["signed_url_status"] == "degraded"
    assert degraded["sources"][0]["timeline_assets"]["signed_url_reason"] == "RuntimeError"
