from types import SimpleNamespace

import pytest

import cutsell_worker.uploads as uploads


class FakeS3:
    def __init__(self):
        self.calls = []

    def generate_presigned_post(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "url": "https://example-upload.invalid",
            "fields": {"key": kwargs["Key"], "Content-Type": kwargs["Fields"]["Content-Type"]},
        }


def _config():
    return SimpleNamespace(s3_bucket="cutsell-bucket", aws_region="us-east-1")


def test_mobile_upload_is_bounded_and_returns_product_source_uri(monkeypatch):
    monkeypatch.setattr(uploads, "load_runtime_config", _config)
    monkeypatch.delenv("CUTSELL_UPLOAD_PREFIX", raising=False)
    client = FakeS3()
    result = uploads.create_presigned_upload(
        project_id="project-1",
        user_id="user-1",
        original_name="My raw take.MOV",
        content_type="video/quicktime",
        size_bytes=12_345_678,
        client=client,
    )
    assert result["method"] == "POST"
    assert result["source_uri"].startswith("s3://cutsell-bucket/cutsell/uploads/")
    assert result["source_uri"].endswith("-My-raw-take.mov")
    call = client.calls[0]
    assert call["Bucket"] == "cutsell-bucket"
    assert call["Fields"]["Content-Type"] == "video/quicktime"
    assert ["content-length-range", 1, 12_345_678] in call["Conditions"]


def test_mobile_upload_rejects_non_video_extension(monkeypatch):
    monkeypatch.setattr(uploads, "load_runtime_config", _config)
    with pytest.raises(ValueError, match="unsupported video extension"):
        uploads.create_presigned_upload(
            project_id="project-1",
            user_id="user-1",
            original_name="payload.exe",
            content_type="application/octet-stream",
            size_bytes=100,
            client=FakeS3(),
        )


def test_product_source_uri_must_use_configured_bucket_and_prefix(monkeypatch):
    monkeypatch.setattr(uploads, "load_runtime_config", _config)
    monkeypatch.delenv("CUTSELL_UPLOAD_PREFIX", raising=False)
    assert uploads.validate_product_source_uri(
        "s3://cutsell-bucket/cutsell/uploads/u/p/video.mp4"
    ) == ("cutsell-bucket", "cutsell/uploads/u/p/video.mp4")
    with pytest.raises(ValueError, match="bucket"):
        uploads.validate_product_source_uri("s3://other-bucket/cutsell/uploads/u/p/video.mp4")
    with pytest.raises(ValueError, match="outside"):
        uploads.validate_product_source_uri("s3://cutsell-bucket/Editdna bloopers videos/video.mp4")


def test_mobile_upload_rejects_oversize_file(monkeypatch):
    monkeypatch.setattr(uploads, "load_runtime_config", _config)
    with pytest.raises(ValueError, match="outside allowed range"):
        uploads.create_presigned_upload(
            project_id="project-1",
            user_id="user-1",
            original_name="video.mp4",
            content_type="video/mp4",
            size_bytes=uploads.MAX_UPLOAD_BYTES + 1,
            client=FakeS3(),
        )
