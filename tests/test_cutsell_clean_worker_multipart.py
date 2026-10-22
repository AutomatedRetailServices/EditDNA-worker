import json
from types import SimpleNamespace

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
    # Build 22: the session stays (24 h) as completed -- a repeated complete is not a 404.
    again = multipart.complete_multipart_upload(
        upload_id="upload-123", user_id="user-1", project_id="project-1",
        parts=[{"part_number": 1, "etag": '"etag-1"'}], s3=s3, redis_client=redis,
    )
    assert again == completed and len(s3.completed) == 1
    status = multipart.list_multipart_parts(
        upload_id="upload-123", user_id="user-1", project_id="project-1", s3=s3, redis_client=redis,
    )
    assert status["state"] == "completed" and status["source_uri"] == completed["source_uri"]
    try:
        multipart.presign_multipart_part(upload_id="upload-123", user_id="user-1", project_id="project-1",
                                         part_number=1, s3=s3, redis_client=redis)
        raise AssertionError("a completed upload must not sign parts")
    except ValueError as exc:
        assert "already completed" in str(exc)


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


def test_s3_client_uses_configured_region_when_none_given(monkeypatch):
    # Part signing/listing/completion pass no region; they must follow AWS_REGION
    # (e.g. "auto" for Cloudflare R2) instead of silently signing for us-east-1.
    monkeypatch.setenv("AWS_REGION", "auto")
    assert multipart._s3_client().meta.region_name == "auto"
    assert multipart._s3_client(region="eu-west-1").meta.region_name == "eu-west-1"


def test_s3_client_falls_back_to_us_east_1_without_config(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(multipart, "load_runtime_config",
                        lambda: type("C", (), {"aws_region": None})())
    assert multipart._s3_client().meta.region_name == "us-east-1"


def test_part_url_is_sigv4_for_r2_endpoint(monkeypatch):
    # R2 rejects the legacy us-east-1 query signature (401); with the configured
    # region the part URL is SigV4 and accepted. Offline: presigning needs no network.
    monkeypatch.setenv("AWS_REGION", "auto")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    redis = FakeRedis()
    session = {"upload_id": "up-1", "user_id": "u", "project_id": "p", "bucket": "cutsell-videos",
               "object_key": "cutsell/uploads/u/p/video.mov", "part_count": 2}
    redis.set(multipart._session_key("up-1"), json.dumps(session))
    out = multipart.presign_multipart_part(upload_id="up-1", user_id="u", project_id="p",
                                           part_number=1, redis_client=redis)
    assert out["upload_url"].startswith("https://example.r2.cloudflarestorage.com/")
    assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in out["upload_url"]
    assert "/auto/s3/aws4_request" in out["upload_url"].replace("%2F", "/")


def test_routes_accept_optional_part_size_and_link_life():
    from cutsell_app.multipart_routes import MultipartSignRequest, MultipartStartRequest
    import pydantic

    base = {"project_id": "p", "user_id": "u", "original_name": "a.mp4", "size_bytes": 10}
    assert MultipartStartRequest(**base).part_size is None
    assert MultipartStartRequest(**base, part_size=8 * 1024 * 1024).part_size == 8 * 1024 * 1024
    for bad in (1024, 65 * 1024 * 1024):
        try:
            MultipartStartRequest(**base, part_size=bad)
        except pydantic.ValidationError:
            continue
        raise AssertionError(bad)
    assert MultipartSignRequest(project_id="p", user_id="u").expires_in is None
    assert MultipartSignRequest(project_id="p", user_id="u", expires_in=3600).expires_in == 3600


def test_r2_refusal_reaches_the_app_with_its_code(monkeypatch):
    redis = FakeRedis()

    class Refusing(FakeS3):
        def complete_multipart_upload(self, **kwargs):
            exc = Exception("boom")
            exc.response = {"Error": {"Code": "InvalidPart", "Message": "One or more of the specified parts could not be found."}}
            raise exc

    s3 = Refusing()
    monkeypatch.setattr(multipart, "prepare_upload_target", lambda **kwargs: _target(kwargs["size_bytes"]))
    multipart.start_multipart_upload(project_id="p", user_id="u", original_name="v.mov", content_type="video/quicktime",
                                     size_bytes=20 * 1024 * 1024, s3=s3, redis_client=redis)
    monkeypatch.setattr(routes, "complete_multipart_upload",
                        lambda **kw: multipart.complete_multipart_upload(**kw, s3=s3, redis_client=redis))
    client = TestClient(app)
    response = client.post("/v1/uploads/multipart/upload-123/complete",
                           json={"project_id": "p", "user_id": "u",
                                 "parts": [{"part_number": 1, "etag": "a"}, {"part_number": 2, "etag": "b"}]})
    assert response.status_code == 502
    assert "InvalidPart" in response.json()["detail"] and "CompleteMultipartUpload" in response.json()["detail"]
    # Not completed: the session is still open for a retry.
    assert "completed" not in json.loads(list(redis.data.values())[0])


def test_part_links_reuse_one_s3_client_and_one_redis(monkeypatch):
    """22 Oct: a new client per part link made ~80 links take ~50 s; the phone gave up at 20 s."""
    monkeypatch.setattr(multipart, "load_runtime_config", lambda: SimpleNamespace(aws_region="auto", redis_url="redis://x:6379/0"))
    multipart._cached_s3.cache_clear()
    multipart._cached_redis.cache_clear()
    assert multipart._s3_client() is multipart._s3_client()
    assert multipart._s3_client(region="eu-west-1") is not multipart._s3_client()
    assert multipart._redis_client() is multipart._redis_client()


def test_a_three_gb_iphone_video_can_start():
    from cutsell_worker import uploads
    assert uploads.MAX_UPLOAD_BYTES >= 3 * 1024 ** 3
