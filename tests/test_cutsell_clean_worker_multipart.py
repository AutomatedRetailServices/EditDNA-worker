import json

from fastapi.testclient import TestClient

import cutsell_app.multipart_routes as routes
from cutsell_app.main import app
import cutsell_worker.multipart_uploads as multipart


class FakeRedis:
    def __init__(self):
        self.data = {}
    def set(self, key, value, ex=None):
        self.data[key] = value
        return True
    def get(self, key):
        return self.data.get(key)
    def delete(self, key):
        self.data.pop(key, None)
        return 1


class FakeS3:
    def __init__(self):
        self.completed = []
        self.aborted = []
    def create_multipart_upload(self, **kwargs):
        self.created = kwargs
        return {"UploadId": "upload-123"}
    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.presigned = (operation, Params, ExpiresIn)
        return "https://upload.invalid/part"
    def list_parts(self, **kwargs):
        return {
            "Parts": [
                {"PartNumber": 2, "ETag": '"etag-2"', "Size": 10},
                {"PartNumber": 1, "ETag": '"etag-1"', "Size": 16 * 1024 * 1024},
            ]
        }
    def complete_multipart_upload(self, **kwargs):
        self.completed.append(kwargs)
        return {"ETag": '"final"'}
    def abort_multipart_upload(self, **kwargs):
        self.aborted.append(kwargs)
        return {}


def _target(size_bytes=20 * 1024 * 1024):
    return {
        "bucket": "bucket",
        "region": "us-east-1",
        "object_key": "cutsell/uploads/u/p/video.mov",
        "source_uri": "s3://bucket/cutsell/uploads/u/p/video.mov",
        "content_type": "video/quicktime",
        "size_bytes": size_bytes,
    }


def test_multipart_session_can_resume_sign_and_complete(monkeypatch):
    redis = FakeRedis()
    s3 = FakeS3()
    monkeypatch.setattr(multipart, "prepare_upload_target", lambda **kwargs: _target(kwargs["size_bytes"]))
    started = multipart.start_multipart_upload(
        project_id="project-1",
        user_id="user-1",
        original_name="video.mov",
        content_type="video/quicktime",
        size_bytes=20 * 1024 * 1024,
        s3=s3,
        redis_client=redis,
    )
    assert started["upload_id"] == "upload-123"
    assert started["part_count"] == 2
    assert started["part_size"] == 16 * 1024 * 1024

    signed = multipart.presign_multipart_part(
        upload_id="upload-123",
        user_id="user-1",
        project_id="project-1",
        part_number=2,
        s3=s3,
        redis_client=redis,
    )
    assert signed["part_number"] == 2
    assert s3.presigned[0] == "upload_part"

    resumed = multipart.list_multipart_parts(
        upload_id="upload-123",
        user_id="user-1",
        project_id="project-1",
        s3=s3,
        redis_client=redis,
    )
    assert resumed["uploaded_part_numbers"] == [1, 2]

    completed = multipart.complete_multipart_upload(
        upload_id="upload-123",
        user_id="user-1",
        project_id="project-1",
        parts=[
            {"part_number": 2, "etag": '"etag-2"'},
            {"part_number": 1, "etag": '"etag-1"'},
        ],
        s3=s3,
        redis_client=redis,
    )
    assert completed["state"] == "uploaded"
    assert completed["source_uri"].startswith("s3://bucket/cutsell/uploads/")
    assert s3.completed[0]["MultipartUpload"]["Parts"][0]["PartNumber"] == 1
    assert redis.data == {}


def test_multipart_session_rejects_wrong_owner(monkeypatch):
    redis = FakeRedis()
    s3 = FakeS3()
    monkeypatch.setattr(multipart, "prepare_upload_target", lambda **kwargs: _target(kwargs["size_bytes"]))
    multipart.start_multipart_upload(
        project_id="project-1",
        user_id="user-1",
        original_name="video.mov",
        content_type="video/quicktime",
        size_bytes=6 * 1024 * 1024,
        s3=s3,
        redis_client=redis,
    )
    try:
        multipart.list_multipart_parts(
            upload_id="upload-123",
            user_id="user-2",
            project_id="project-1",
            s3=s3,
            redis_client=redis,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("multipart session must be user scoped")


def test_multipart_complete_requires_every_expected_part(monkeypatch):
    redis = FakeRedis()
    s3 = FakeS3()
    monkeypatch.setattr(multipart, "prepare_upload_target", lambda **kwargs: _target(kwargs["size_bytes"]))
    multipart.start_multipart_upload(
        project_id="project-1",
        user_id="user-1",
        original_name="video.mov",
        content_type="video/quicktime",
        size_bytes=20 * 1024 * 1024,
        s3=s3,
        redis_client=redis,
    )
    try:
        multipart.complete_multipart_upload(
            upload_id="upload-123",
            user_id="user-1",
            project_id="project-1",
            parts=[{"part_number": 1, "etag": '"etag-1"'}],
            s3=s3,
            redis_client=redis,
        )
    except ValueError as exc:
        assert "every expected part" in str(exc)
    else:
        raise AssertionError("incomplete multipart upload must not complete")


def test_abort_multipart_removes_session(monkeypatch):
    redis = FakeRedis()
    s3 = FakeS3()
    monkeypatch.setattr(multipart, "prepare_upload_target", lambda **kwargs: _target(kwargs["size_bytes"]))
    multipart.start_multipart_upload(
        project_id="project-1",
        user_id="user-1",
        original_name="video.mov",
        content_type="video/quicktime",
        size_bytes=6 * 1024 * 1024,
        s3=s3,
        redis_client=redis,
    )
    result = multipart.abort_multipart_upload(
        upload_id="upload-123",
        user_id="user-1",
        project_id="project-1",
        s3=s3,
        redis_client=redis,
    )
    assert result["state"] == "canceled"
    assert len(s3.aborted) == 1
    assert redis.data == {}


def test_multipart_api_routes_are_mobile_friendly(monkeypatch):
    monkeypatch.setattr(
        routes,
        "start_multipart_upload",
        lambda **kwargs: {
            "upload_id": "upload-123",
            "project_id": kwargs["project_id"],
            "user_id": kwargs["user_id"],
            "source_uri": "s3://bucket/cutsell/uploads/u/p/video.mov",
            "object_key": "cutsell/uploads/u/p/video.mov",
            "content_type": "video/quicktime",
            "size_bytes": kwargs["size_bytes"],
            "part_size": 16 * 1024 * 1024,
            "part_count": 2,
            "created_at": "2026-08-07T00:00:00+00:00",
            "expires_in": 86400,
            "schema_version": "cutsell.multipart.v1",
            "bucket": "bucket",
        },
    )
    client = TestClient(app)
    response = client.post("/v1/uploads/multipart/start", json={
        "project_id": "project-1",
        "user_id": "user-1",
        "original_name": "video.mov",
        "content_type": "video/quicktime",
        "size_bytes": 20 * 1024 * 1024,
    })
    assert response.status_code == 200
    assert response.json()["upload_id"] == "upload-123"
    assert response.json()["part_count"] == 2
