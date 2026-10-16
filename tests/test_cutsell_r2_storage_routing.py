"""R2 move: reads of old cuts keep working from the old Amazon bucket (FALLBACK_*), every write
goes to the current storage, and with no FALLBACK_* settings nothing changes."""
from __future__ import annotations

from pathlib import Path

import pytest

from cutsell_worker import overlay_uploads, sfx_library, storage, storage_routing, timeline_asset_s3, uploads

CURRENT, OLD = "cutsell-r2", "script2clipshop-video-old"
SCOPE = uploads.scoped_upload_prefix(user_id="u", project_id="p")
OVERLAY_SCOPE = overlay_uploads.overlay_prefix(user_id="u", project_id="p")


class NotFound(Exception):
    def __init__(self):
        super().__init__("404")
        self.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}


class Forbidden(Exception):
    def __init__(self):
        super().__init__("403")
        self.response = {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}


class FakeClient:
    def __init__(self, name, objects=(), fail=None):
        self.name, self.objects, self.fail, self.calls = name, set(objects), fail, []

    def _get(self, bucket, key):
        if self.fail is not None:
            raise self.fail
        if (bucket, key) not in self.objects:
            raise NotFound()

    def download_file(self, bucket, key, path):
        self.calls.append(("download_file", bucket, key))
        self._get(bucket, key)
        Path(path).write_text(f"{self.name}:{bucket}/{key}")

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", Bucket, Key))
        self._get(Bucket, Key)
        return {"ContentLength": 1, "from": self.name}

    def generate_presigned_url(self, op, Params, ExpiresIn):
        self.calls.append((op, Params["Bucket"], Params["Key"]))
        return f"https://{self.name}/{Params['Bucket']}/{Params['Key']}"

    def generate_presigned_post(self, **kw):
        self.calls.append(("generate_presigned_post", kw["Bucket"], kw["Key"]))
        return {"url": f"https://{self.name}/{kw['Bucket']}", "fields": {"key": kw["Key"]}}

    def upload_file(self, source, bucket, key, ExtraArgs=None):
        self.calls.append(("upload_file", bucket, key))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", CURRENT)
    monkeypatch.setenv("AWS_REGION", "auto")
    for name in storage_routing.FALLBACK_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def amazon(env):
    """The old Amazon settings configured, its client replaced by a fake (records how it was built)."""
    env.setenv("FALLBACK_S3_BUCKET", OLD)
    env.setenv("FALLBACK_AWS_REGION", "us-east-2")
    env.setenv("FALLBACK_AWS_ACCESS_KEY_ID", "fake-id")
    env.setenv("FALLBACK_AWS_SECRET_ACCESS_KEY", "fake-secret")
    client = FakeClient("amazon", objects={(OLD, f"{SCOPE}old.mp4"), (OLD, f"{SCOPE}moved.mp4"),
                                           (OLD, f"{OVERLAY_SCOPE}p.png"), (OLD, "assets/sfx/v1/tap_select.m4a")})
    built = []
    env.setattr(storage_routing, "fallback_client_factory", lambda fb: built.append(fb) or client)
    client.built = built
    return client


def test_fallback_needs_bucket_and_credentials(env):
    assert storage_routing.fallback_storage() is None
    env.setenv("FALLBACK_S3_BUCKET", OLD)
    assert storage_routing.fallback_storage() is None  # never send the R2 keys to Amazon
    env.setenv("FALLBACK_AWS_ACCESS_KEY_ID", "a")
    env.setenv("FALLBACK_AWS_SECRET_ACCESS_KEY", "b")
    fb = storage_routing.fallback_storage()
    assert fb.bucket == OLD and fb.region == "us-east-1" and fb.endpoint_url == "https://s3.us-east-1.amazonaws.com"


def test_amazon_client_uses_an_explicit_amazon_endpoint_even_with_an_r2_endpoint_set(env):
    env.setenv("AWS_ENDPOINT_URL_S3", "https://acct.r2.cloudflarestorage.com")
    fb = storage_routing.FallbackStorage(OLD, "us-east-2", "fake-id", "fake-secret")
    client = storage_routing._make_fallback_client(fb)
    assert client.meta.endpoint_url == "https://s3.us-east-2.amazonaws.com"
    assert client.meta.region_name == "us-east-2"
    assert storage_routing.default_client(region_name="auto").meta.endpoint_url == "https://acct.r2.cloudflarestorage.com"


def test_without_fallback_reads_are_exactly_the_old_single_call(env, tmp_path):
    r2 = FakeClient("r2", objects={(CURRENT, "k.mp4")})
    storage.download_source(f"s3://{CURRENT}/k.mp4", str(tmp_path / "a"), client=r2)
    url = storage_routing.presigned_get(CURRENT, "k.mp4", expires_in=60, client=r2)
    assert r2.calls == [("download_file", CURRENT, "k.mp4"), ("get_object", CURRENT, "k.mp4")]  # no extra HEAD
    assert url == f"https://r2/{CURRENT}/k.mp4"
    with pytest.raises(NotFound):
        storage.download_source(f"s3://{CURRENT}/missing.mp4", str(tmp_path / "b"), client=r2)
    with pytest.raises(ValueError, match="bucket"):
        uploads.validate_product_source_uri(f"s3://{OLD}/{SCOPE}old.mp4", project_id="p", user_id="u")


def test_old_bucket_uri_is_read_from_amazon(amazon, tmp_path):
    r2 = FakeClient("r2")
    out = storage.download_source(f"s3://{OLD}/{SCOPE}old.mp4", str(tmp_path / "a"), client=r2)
    assert Path(out).read_text() == f"amazon:{OLD}/{SCOPE}old.mp4"
    assert r2.calls == []
    assert amazon.built[0].endpoint_url == "https://s3.us-east-2.amazonaws.com"
    url = uploads.create_presigned_source_download(f"s3://{OLD}/{SCOPE}old.mp4", project_id="p", user_id="u", client=r2)
    assert url["url"] == f"https://amazon/{OLD}/{SCOPE}old.mp4" and r2.calls == []


def test_current_bucket_miss_falls_back_to_amazon(amazon, tmp_path):
    r2 = FakeClient("r2", objects={(CURRENT, f"{SCOPE}new.mp4")})
    moved = storage.download_source(f"s3://{CURRENT}/{SCOPE}moved.mp4", str(tmp_path / "m"), client=r2)
    assert Path(moved).read_text() == f"amazon:{OLD}/{SCOPE}moved.mp4"
    fresh = storage.download_source(f"s3://{CURRENT}/{SCOPE}new.mp4", str(tmp_path / "n"), client=r2)
    assert Path(fresh).read_text() == f"r2:{CURRENT}/{SCOPE}new.mp4"
    assert storage_routing.presigned_get(CURRENT, f"{SCOPE}moved.mp4", expires_in=60, client=r2) == \
        f"https://amazon/{OLD}/{SCOPE}moved.mp4"
    assert storage_routing.presigned_get(CURRENT, f"{SCOPE}new.mp4", expires_in=60, client=r2) == \
        f"https://r2/{CURRENT}/{SCOPE}new.mp4"
    assert storage_routing.head(CURRENT, f"{SCOPE}moved.mp4", client=r2)["from"] == "amazon"
    # a miss in both is still an error; a non-404 error on R2 is never hidden by the fallback
    with pytest.raises(NotFound):
        storage.download_source(f"s3://{CURRENT}/{SCOPE}nowhere.mp4", str(tmp_path / "x"), client=r2)
    denied = FakeClient("r2", fail=Forbidden())
    with pytest.raises(Forbidden):
        storage.download_source(f"s3://{CURRENT}/{SCOPE}moved.mp4", str(tmp_path / "y"), client=denied)


def test_writes_never_go_to_amazon(amazon, env):
    env.setenv("AWS_ENDPOINT_URL_S3", "https://acct.r2.cloudflarestorage.com")
    r2 = FakeClient("r2")
    out = uploads.create_presigned_upload(project_id="p", user_id="u", original_name="v.mp4", content_type="video/mp4",
                                          size_bytes=10, client=r2)
    assert out["source_uri"].startswith(f"s3://{CURRENT}/") and out["method"] == "PUT"
    over = overlay_uploads.create_overlay_presigned_upload(project_id="p", user_id="u", original_name="a.png",
                                                           content_type="image/png", size_bytes=10, client=r2)
    assert over["uri"].startswith(f"s3://{CURRENT}/")
    assert amazon.calls == [] and amazon.built == []
    assert all(call[1] == CURRENT for call in r2.calls)


def test_timeline_persister_writes_to_the_current_bucket(amazon, tmp_path):
    r2 = FakeClient("r2")
    source = tmp_path / "vo.m4a"
    source.write_bytes(b"x")
    uri = timeline_asset_s3.s3_persister(user_id="u", project_id="p", client=r2)(str(source), "a/b/asset1.m4a")
    assert uri.startswith(f"s3://{CURRENT}/") and r2.calls[0][0:2] == ("upload_file", CURRENT)
    assert amazon.calls == []


def test_validation_accepts_the_fallback_bucket_for_reads_with_the_same_scope_checks(amazon):
    assert uploads.validate_product_source_uri(f"s3://{OLD}/{SCOPE}old.mp4", project_id="p", user_id="u") == \
        (OLD, f"{SCOPE}old.mp4")
    with pytest.raises(ValueError, match="outside"):  # tenant scope as strict as ever
        uploads.validate_product_source_uri(f"s3://{OLD}/{uploads.scoped_upload_prefix(user_id='x', project_id='p')}a.mp4",
                                            project_id="p", user_id="u")
    with pytest.raises(ValueError, match="bucket"):  # a foreign bucket that is not the fallback
        uploads.validate_product_source_uri(f"s3://someone-else/{SCOPE}old.mp4", project_id="p", user_id="u")
    assert overlay_uploads.validate_overlay_uri(f"s3://{OLD}/{OVERLAY_SCOPE}p.png", user_id="u", project_id="p")[0] == OLD
    with pytest.raises(ValueError, match="bucket"):
        overlay_uploads.validate_overlay_uri(f"s3://someone-else/{OVERLAY_SCOPE}p.png", user_id="u", project_id="p")
    link = overlay_uploads.create_presigned_overlay_download(f"s3://{OLD}/{OVERLAY_SCOPE}p.png", user_id="u",
                                                            project_id="p", client=FakeClient("r2"))
    assert link["url"] == f"https://amazon/{OLD}/{OVERLAY_SCOPE}p.png"
    # the timeline fetcher reads old-bucket references; the bucket check still refuses others
    with pytest.raises(RuntimeError):
        timeline_asset_s3._checked_key(f"s3://someone-else/{SCOPE}x.m4a", user_id="u", project_id="p", for_read=True)
    with pytest.raises(RuntimeError):  # never for a write
        timeline_asset_s3._checked_key(f"s3://{OLD}/{SCOPE}x.m4a", user_id="u", project_id="p")
    assert timeline_asset_s3._checked_key(f"s3://{OLD}/{SCOPE}x.m4a", user_id="u", project_id="p", for_read=True)[0] == OLD


def test_sounds_follow_the_current_storage_and_fall_back(amazon, env, tmp_path):
    from cutsell_worker import export_job

    r2 = FakeClient("r2", objects={(CURRENT, "assets/sfx/v1/good.m4a")})
    env.setattr(storage_routing, "default_client", lambda region_name=None: r2)
    assert sfx_library.sfx_uri("tap") == f"s3://{CURRENT}/assets/sfx/v1/tap_select.m4a"
    clip = type("Clip", (), {"sfx_id": "tap", "audio_id": "a1"})()
    out = export_job._resolve_sound_effect(clip, directory=str(tmp_path), index=0)
    assert Path(out).read_text() == f"amazon:{OLD}/assets/sfx/v1/tap_select.m4a"  # not on R2 yet
    clip.sfx_id = "good"
    out = export_job._resolve_sound_effect(clip, directory=str(tmp_path), index=1)
    assert Path(out).read_text() == f"r2:{CURRENT}/assets/sfx/v1/good.m4a"
    manifest = sfx_library.signed_manifest(client=_ManifestClient(r2))
    urls = {row["id"]: row["url"] for row in manifest["sounds"]}
    assert urls["good"].startswith("https://r2/") and urls["tap"].startswith("https://amazon/")


class _ManifestClient:
    """R2 client without a stored manifest.json (and without the sounds except "good")."""

    def __init__(self, inner):
        self.inner = inner

    def get_object(self, Bucket, Key):
        raise NotFound()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_default_client_signs_links_with_v4_in_the_configured_region(monkeypatch):
    # Live finding (16 Oct 2026, after the R2 move): a client made without a region signed
    # presigned GET links with the legacy V2 scheme, and R2 answered 401 to the Clean up link.
    from urllib.parse import urlparse, parse_qs
    from cutsell_worker import storage_routing
    monkeypatch.setenv("AWS_REGION", "auto")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    url = storage_routing.default_client().generate_presigned_url(
        "get_object", Params={"Bucket": "cutsell-videos", "Key": "a/b.m4a"}, ExpiresIn=60)
    query = parse_qs(urlparse(url).query)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Credential"][0].split("/")[2] == "auto"
    assert urlparse(url).hostname.endswith("r2.cloudflarestorage.com")
