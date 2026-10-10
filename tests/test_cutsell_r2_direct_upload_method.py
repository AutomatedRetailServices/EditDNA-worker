"""Direct uploads follow the storage: POST form on Amazon S3 (unchanged), signed PUT on
an S3-compatible endpoint such as Cloudflare R2 (which answers 501 to POST forms)."""
from urllib.parse import parse_qs, urlparse

import pytest

from cutsell_worker import overlay_uploads, uploads
from cutsell_worker.timeline_upload_registration import register_timeline_upload


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def get(self, key):
        return self.data.get(key)


@pytest.fixture
def s3_env(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "cutsell-videos")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return monkeypatch


@pytest.fixture
def r2_env(s3_env):
    s3_env.setenv("AWS_ENDPOINT_URL_S3", "https://example.r2.cloudflarestorage.com")
    s3_env.setenv("AWS_REGION", "auto")
    return s3_env


def _calls():
    kw = dict(project_id="p", user_id="u")
    return {
        "video": lambda: uploads.create_presigned_upload(original_name="clip.mp4", content_type="video/mp4",
                                                         size_bytes=1234, **kw),
        "voice": lambda: uploads.create_presigned_voice_over_upload(original_name="vo.m4a", content_type="audio/mp4",
                                                                    size_bytes=1234, **kw),
        "overlay": lambda: overlay_uploads.create_overlay_presigned_upload(original_name="f.png",
                                                                           content_type="image/png",
                                                                           size_bytes=1234, **kw),
    }


@pytest.mark.parametrize("name", ["video", "voice", "overlay"])
def test_amazon_s3_keeps_post_form(s3_env, name):
    out = _calls()[name]()
    assert out["method"] == "POST"
    assert out["fields"]["Content-Type"]
    assert out["headers"] == {}


@pytest.mark.parametrize("name,ctype", [("video", "video/mp4"), ("voice", "audio/mp4"), ("overlay", "image/png")])
def test_r2_gets_signed_put_with_exact_type_and_size(r2_env, name, ctype):
    out = _calls()[name]()
    assert out["method"] == "PUT"
    assert out["fields"] == {}
    assert out["headers"] == {"Content-Type": ctype}
    url = urlparse(out["upload_url"])
    assert url.netloc == "example.r2.cloudflarestorage.com"
    query = parse_qs(url.query)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    signed = query["X-Amz-SignedHeaders"][0].split(";")
    assert "content-type" in signed and "content-length" in signed


def test_timeline_upload_passes_method_and_headers_to_app(r2_env):
    out = register_timeline_upload(project_id="p", user_id="u", media_class="voice_over",
                                   original_name="vo.m4a", content_type="audio/mp4", size_bytes=99,
                                   redis_client=FakeRedis())
    assert out["method"] == "PUT"
    assert out["headers"] == {"Content-Type": "audio/mp4"}
    assert out["fields"] == {}
