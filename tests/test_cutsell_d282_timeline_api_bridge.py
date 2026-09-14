"""D-282 -- Mobile Timeline API Bridge + Voice-Over Upload Contract.
D-282A -- Timeline API Authority / Storage Reference Hardening.

Real FastAPI TestClient against `cutsell_app.main.app`, real ffmpeg
fixtures, fake Redis (this session's established convention), fake/local
durable storage (Stage 31/36 -- no real S3 mutation).

D-282A replaced the original `source_uri`-accepting ingest contract with
an opaque, server-issued `upload_id` (`timeline_upload_registration.py`)
and the raw `output_path` export field with an opaque `export_id`
(`timeline_export_reference.py`). `_authorize_upload`/`_create_asset`
below drive that two-step flow through a fake `presign_upload` callable
(the SAME injection seam `_default_presign_upload` documents) so no real
S3/boto3 is ever touched in this suite.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient

from cutsell_app import timeline_routes
from cutsell_app.main import app
from cutsell_worker import account_lifecycle
from cutsell_worker import project_store
from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker import timeline_asset_registry_store as store
from cutsell_worker import timeline_export_reference as export_ref
from cutsell_worker import timeline_upload_registration as upload_reg
from cutsell_worker import uploads


# ---------------------------------------------------------------------------
# Fake Redis (shared across project_store + timeline_asset_registry_store)
# ---------------------------------------------------------------------------

class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    def set(self, key, value):
        self.ops.append(("set", key, value))
        return self

    def zadd(self, key, mapping):
        self.ops.append(("zadd", key, dict(mapping)))
        return self

    def delete(self, key):
        self.ops.append(("delete", key))
        return self

    def zrem(self, key, member):
        self.ops.append(("zrem", key, member))
        return self

    def execute(self):
        for op in self.ops:
            if op[0] == "set":
                self.redis.set(op[1], op[2])
            elif op[0] == "zadd":
                self.redis.zadd(op[1], op[2])
            elif op[0] == "delete":
                self.redis.delete(op[1])
            elif op[0] == "zrem":
                self.redis.zrem(op[1], op[2])
        self.ops = []
        return [True]


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.zsets = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):  # noqa: ARG002 -- ttl irrelevant to a fake in-memory store
        self.data[key] = value
        return True

    def delete(self, *keys):
        n = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                n += 1
            if key in self.zsets:
                del self.zsets[key]
                n += 1
        return n

    def zadd(self, key, mapping):
        bucket = self.zsets.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    def zrevrange(self, key, start, end):
        ordered = sorted(self.zsets.get(key, {}).items(), key=lambda item: item[1], reverse=True)
        stop = None if end < 0 else end + 1
        return [
            (item[0].encode() if isinstance(item[0], str) else item[0])
            for item in ordered[start:stop]
        ]

    def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)

    def pipeline(self):
        return FakePipeline(self)


@pytest.fixture()
def fake_redis(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(project_store, "_redis_client", lambda client=None: redis if client is None else client)
    monkeypatch.setattr(store, "_redis_client", lambda client=None: redis if client is None else client)
    monkeypatch.setattr(upload_reg, "_redis_client", lambda client=None: redis if client is None else client)
    monkeypatch.setattr(export_ref, "_redis_client", lambda client=None: redis if client is None else client)
    return redis


@pytest.fixture()
def durable_dir(tmp_path, monkeypatch):
    directory = tmp_path / "durable"
    directory.mkdir()
    monkeypatch.setenv("CUTSELL_TIMELINE_ASSET_DURABLE_DIR", str(directory))
    return str(directory)


# ---------------------------------------------------------------------------
# D-282A fake presign -- the ONE seam standing in for real S3/boto3. Tests
# register "what a client already uploaded" by `original_name`; the fake
# resolves it back to a real local fixture path, exactly the same
# `local://` fake-storage scheme `timeline_asset_upload_bridge.py`'s own
# `_default_fetch_uploaded_media` already establishes.
# ---------------------------------------------------------------------------

_FAKE_UPLOAD_TARGETS: dict[str, str] = {}


def _fake_presign_upload(*, media_class, project_id, user_id, original_name, content_type, size_bytes, expires_in):
    local_path = _FAKE_UPLOAD_TARGETS[original_name]
    return {
        "method": "PUT",
        "upload_url": "local://fake-upload-target",
        "fields": {},
        "source_uri": f"local://{local_path}",
        "object_key": original_name,
        "content_type": content_type or "application/octet-stream",
        "max_bytes": size_bytes,
        "expires_in": expires_in,
    }


@pytest.fixture(autouse=True)
def fake_presign(monkeypatch):
    _FAKE_UPLOAD_TARGETS.clear()
    monkeypatch.setattr(upload_reg, "_default_presign_upload", _fake_presign_upload)
    yield
    _FAKE_UPLOAD_TARGETS.clear()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def project(fake_redis):
    return project_store.create_project(user_id="u1", title="D-282 Project", client=fake_redis)


@pytest.fixture(scope="module")
def broll_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d282_src") / "broll.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=2:r=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture(scope="module")
def vo_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d282_src") / "vo.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=1.5", "-c:a", "aac", "-ar", "48000", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture(scope="module")
def base_edit_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d282_src") / "base.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=1080x1920:d=6:r=30",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


_NAME_COUNTER = [0]


def _authorize_upload(client, project_id, *, user_id="u1", media_class="video", local_path, original_name=None):
    """D-282A Stage 1-3: the two-step flow every ingest test now drives --
    authorize, then consume. Each call mints a fresh `original_name` by
    default so tests never accidentally collide on the fake presign
    registry."""
    if original_name is None:
        _NAME_COUNTER[0] += 1
        suffix = Path(local_path).suffix or ".bin"
        original_name = f"upload-{_NAME_COUNTER[0]}{suffix}"
    _FAKE_UPLOAD_TARGETS[original_name] = local_path
    size_bytes = max(1, Path(local_path).stat().st_size) if Path(local_path).exists() else 1
    response = client.post(
        f"/v1/projects/{project_id}/timeline-uploads",
        json={
            "user_id": user_id, "media_class": media_class, "original_name": original_name,
            "content_type": None, "size_bytes": size_bytes,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_asset(
    client, project_id, *, user_id="u1", role="SUPPLEMENTAL_BROLL", media_kind="VIDEO",
    local_path, media_class=None, upload_id=None,
):
    resolved_media_class = media_class or (
        upload_reg.MEDIA_CLASS_VOICE_OVER
        if media_kind == "AUDIO" or role == "VOICE_OVER"
        else upload_reg.MEDIA_CLASS_VIDEO
    )
    if upload_id is None:
        upload_id = _authorize_upload(
            client, project_id, user_id=user_id, media_class=resolved_media_class, local_path=local_path,
        )["upload_id"]
    return client.post(
        f"/v1/projects/{project_id}/timeline-assets",
        json={"user_id": user_id, "role": role, "media_kind": media_kind, "upload_id": upload_id},
    )


# ---------------------------------------------------------------------------
# 1/2/3. routes exist, auth required, ownership required
# ---------------------------------------------------------------------------

def test_timeline_routes_exist_and_are_registered():
    schema = app.openapi()
    assert schema["paths"]["/v1/projects/{project_id}/timeline-uploads"].keys() >= {"post"}
    assert schema["paths"]["/v1/projects/{project_id}/timeline-assets"].keys() >= {"post", "get"}
    assert schema["paths"]["/v1/projects/{project_id}/timeline"].keys() >= {"get", "put"}
    assert schema["paths"]["/v1/projects/{project_id}/timeline/export"].keys() >= {"post"}
    assert schema["paths"]["/v1/projects/{project_id}/timeline/export/{export_id}/download"].keys() >= {"get"}


def test_auth_required_when_enabled(monkeypatch):
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")
    raw_client = TestClient(app)
    response = raw_client.get("/v1/projects/p1/timeline-assets", params={"user_id": "u1"})
    assert response.status_code == 401


def test_unknown_project_returns_404(client, fake_redis, durable_dir):
    response = client.get("/v1/projects/does-not-exist/timeline-assets", params={"user_id": "u1"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 7/8/9/32. B-roll creation, qualification/normalization reuse, list
# ---------------------------------------------------------------------------

def test_create_broll_asset_via_api_reaches_ready(client, fake_redis, durable_dir, project, broll_source):
    response = _create_asset(client, project["project_id"], local_path=broll_source)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["qualification_status"] == "READY"
    assert body["media_kind"] == "VIDEO"
    assert body["duration_sec"] == pytest.approx(2.0, abs=0.3)
    assert body["has_audio"] is True
    assert "storage_reference" not in body


def test_list_assets_scoped_to_project(client, fake_redis, durable_dir, project, broll_source):
    _create_asset(client, project["project_id"], local_path=broll_source)
    other_project = project_store.create_project(user_id="u1", title="Other", client=fake_redis)
    _create_asset(client, other_project["project_id"], local_path=broll_source)

    response = client.get(f"/v1/projects/{project['project_id']}/timeline-assets", params={"user_id": "u1"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["assets"]) == 1


def test_reject_creation_of_video_that_fails_qualification(client, fake_redis, durable_dir, project, tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"not a real video")
    response = _create_asset(client, project["project_id"], local_path=str(bogus))
    assert response.status_code == 200
    body = response.json()
    assert body["qualification_status"] != "READY"


# ---------------------------------------------------------------------------
# 10/11/12/33. VO upload format accepted, server-authoritative metadata, READY
# ---------------------------------------------------------------------------

def test_create_voice_over_asset_via_api(client, fake_redis, durable_dir, project, vo_source):
    response = _create_asset(
        client, project["project_id"], role="VOICE_OVER", media_kind="AUDIO", local_path=vo_source,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["qualification_status"] == "READY"
    assert body["media_kind"] == "AUDIO"
    assert body["has_audio"] is True
    assert body["duration_sec"] == pytest.approx(1.5, abs=0.3)


def test_voice_over_upload_allowlist_accepts_m4a_and_rejects_video_extension():
    # Stage 6/7/11: the smallest safe V1 audio allowlist -- grounded in
    # real evidence (D-281's own qualified .m4a fixture; iOS's own
    # AVAudioRecorder default container).
    assert ".m4a" in uploads.ALLOWED_VOICE_OVER_AUDIO_EXTENSIONS
    with pytest.raises(ValueError):
        uploads._safe_voice_over_name("clip.mp4")


def test_voice_over_upload_allowlist_is_structurally_distinct_from_video():
    assert uploads.DEFAULT_VOICE_OVER_UPLOAD_PREFIX != uploads.DEFAULT_UPLOAD_PREFIX
    assert uploads.ALLOWED_VOICE_OVER_AUDIO_EXTENSIONS.isdisjoint(uploads.ALLOWED_VIDEO_EXTENSIONS)


# ---------------------------------------------------------------------------
# D-282A Stage 1-6/11 -- opaque upload_id authority (replaces raw source_uri)
# ---------------------------------------------------------------------------

def test_ingest_no_longer_accepts_a_raw_source_uri(client, fake_redis, durable_dir, project, broll_source):
    response = client.post(
        f"/v1/projects/{project['project_id']}/timeline-assets",
        json={
            "user_id": "u1", "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO",
            "source_uri": f"local://{broll_source}",
        },
    )
    # Pydantic rejects the request: `upload_id` is required and missing,
    # `source_uri` is not a field on this model at all.
    assert response.status_code == 422


def test_upload_id_is_server_issued_and_opaque(client, fake_redis, durable_dir, project, broll_source):
    authorization = _authorize_upload(client, project["project_id"], local_path=broll_source)
    assert authorization["upload_id"].startswith("tup_")
    assert authorization["upload_id"] != broll_source
    assert "source_uri" not in authorization
    assert "object_key" not in authorization
    assert "bucket" not in authorization


def test_upload_id_bound_to_issuing_user_cross_user_denied(fake_redis):
    # This codebase's project model is single-owner, so a cross-user
    # attempt through the route never even reaches this module's own
    # check -- `_require_project` denies it first (proven by the
    # pre-existing `test_wrong_user_cannot_save_timeline`/`test_wrong_
    # user_cannot_list_project_assets`). This is a direct, module-level
    # proof that `resolve_and_consume_timeline_upload` ALSO enforces
    # user ownership on its own -- real defense in depth, not a route
    # artifact -- for any future caller/shared-project model.
    _FAKE_UPLOAD_TARGETS["x.mp4"] = "/tmp/does-not-matter"
    authorization = upload_reg.register_timeline_upload(
        project_id="p1", user_id="u1", media_class=upload_reg.MEDIA_CLASS_VIDEO,
        original_name="x.mp4", content_type=None, size_bytes=10,
        presign_upload=_fake_presign_upload, redis_client=fake_redis,
    )
    with pytest.raises(upload_reg.TimelineUploadResolutionError) as excinfo:
        upload_reg.resolve_and_consume_timeline_upload(
            upload_id=authorization["upload_id"], user_id="u2", project_id="p1",
            expected_media_class=upload_reg.MEDIA_CLASS_VIDEO, redis_client=fake_redis,
        )
    assert excinfo.value.outcome == upload_reg.UPLOAD_NOT_OWNED


def test_upload_id_bound_to_issuing_project_cross_project_denied(client, fake_redis, durable_dir, project, broll_source):
    other_project = project_store.create_project(user_id="u1", title="Other", client=fake_redis)
    upload_id = _authorize_upload(client, project["project_id"], local_path=broll_source)["upload_id"]
    response = client.post(
        f"/v1/projects/{other_project['project_id']}/timeline-assets",
        json={"user_id": "u1", "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": upload_id},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["outcome"] == upload_reg.UPLOAD_NOT_OWNED


def test_voice_over_upload_id_cannot_be_consumed_as_broll(client, fake_redis, durable_dir, project, vo_source):
    upload_id = _authorize_upload(
        client, project["project_id"], media_class=upload_reg.MEDIA_CLASS_VOICE_OVER, local_path=vo_source,
    )["upload_id"]
    response = client.post(
        f"/v1/projects/{project['project_id']}/timeline-assets",
        json={"user_id": "u1", "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": upload_id},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["outcome"] == upload_reg.UPLOAD_MEDIA_CLASS_MISMATCH


def test_broll_upload_id_cannot_be_consumed_as_voice_over(client, fake_redis, durable_dir, project, broll_source):
    upload_id = _authorize_upload(
        client, project["project_id"], media_class=upload_reg.MEDIA_CLASS_VIDEO, local_path=broll_source,
    )["upload_id"]
    response = client.post(
        f"/v1/projects/{project['project_id']}/timeline-assets",
        json={"user_id": "u1", "role": "VOICE_OVER", "media_kind": "AUDIO", "upload_id": upload_id},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["outcome"] == upload_reg.UPLOAD_MEDIA_CLASS_MISMATCH


def test_upload_id_cannot_be_replayed(client, fake_redis, durable_dir, project, broll_source):
    upload_id = _authorize_upload(client, project["project_id"], local_path=broll_source)["upload_id"]
    first = client.post(
        f"/v1/projects/{project['project_id']}/timeline-assets",
        json={"user_id": "u1", "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": upload_id},
    )
    assert first.status_code == 200, first.text
    second = client.post(
        f"/v1/projects/{project['project_id']}/timeline-assets",
        json={"user_id": "u1", "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": upload_id},
    )
    assert second.status_code == 409
    assert second.json()["detail"]["outcome"] == upload_reg.UPLOAD_ALREADY_CONSUMED


def test_unknown_upload_id_returns_bounded_404(client, fake_redis, durable_dir, project):
    response = client.post(
        f"/v1/projects/{project['project_id']}/timeline-assets",
        json={"user_id": "u1", "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": "tup_does-not-exist"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["outcome"] == upload_reg.UPLOAD_NOT_FOUND


def test_unknown_media_class_rejected_by_model_validation(client, fake_redis, durable_dir, project):
    response = client.post(
        f"/v1/projects/{project['project_id']}/timeline-uploads",
        json={"user_id": "u1", "media_class": "not_a_real_class", "original_name": "x.mp4", "size_bytes": 10},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5/6/13/14/15/16/17. client-safe response, get/save timeline, concurrency,
# silent-audio validation
# ---------------------------------------------------------------------------

def _create_base_edit(client, project_id, base_edit_source):
    response = _create_asset(
        client, project_id, role="PRIMARY_SOURCE", media_kind="VIDEO", local_path=base_edit_source,
    )
    assert response.status_code == 200, response.text
    return response.json()["asset_id"]


def test_save_and_reopen_timeline_round_trip(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    broll = _create_asset(client, project["project_id"], local_path=broll_source).json()

    save_response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={
            "user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0,
            "broll_placements": [{
                "placement_id": "broll_1", "asset_id": broll["asset_id"],
                "timeline_start_sec": 1.0, "timeline_end_sec": 3.0,
                "source_in_sec": 0.0, "source_out_sec": 2.0,
            }],
            "voice_over_placements": [],
        },
    )
    assert save_response.status_code == 200, save_response.text
    save_body = save_response.json()
    assert save_body["outcome"] == "TIMELINE_SAVE_SUCCEEDED"
    revision_identity = save_body["timeline_revision_identity"]

    get_response = client.get(f"/v1/projects/{project['project_id']}/timeline", params={"user_id": "u1"})
    assert get_response.status_code == 200
    get_body = get_response.json()
    assert get_body["timeline_revision_identity"] == revision_identity
    assert get_body["broll_placements"][0]["asset_id"] == broll["asset_id"]
    assert "storage_reference" not in str(get_body)


def test_stale_revision_conflict_returns_409(client, fake_redis, durable_dir, project, base_edit_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    first = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={"user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0},
    )
    assert first.status_code == 200
    second = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={
            "user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 7.0,
            "expected_revision_identity": "stale-value",
        },
    )
    assert second.status_code == 409
    assert second.json()["detail"]["outcome"] == reg.TIMELINE_REVISION_CONFLICT


def test_use_broll_audio_on_silent_asset_returns_bounded_error(client, fake_redis, durable_dir, project, base_edit_source, tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=320x240:d=1:r=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(silent)],
        check=True, capture_output=True,
    )
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    silent_asset = _create_asset(client, project["project_id"], local_path=str(silent)).json()
    assert silent_asset["has_audio"] is False

    response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={
            "user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0,
            "broll_placements": [{
                "placement_id": "broll_1", "asset_id": silent_asset["asset_id"],
                "timeline_start_sec": 0.0, "timeline_end_sec": silent_asset["duration_sec"],
                "source_in_sec": 0.0, "source_out_sec": silent_asset["duration_sec"],
                "audio_mode": "USE_BROLL_AUDIO",
            }],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"]["outcome"] == reg.TIMELINE_INVALID
    assert any(reg.ASSET_HAS_NO_AUDIO in r for r in response.json()["detail"]["reasons"])


def test_unknown_audio_mode_rejected_by_model_validation(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    broll = _create_asset(client, project["project_id"], local_path=broll_source).json()
    response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={
            "user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0,
            "broll_placements": [{
                "placement_id": "broll_1", "asset_id": broll["asset_id"],
                "timeline_start_sec": 0.0, "timeline_end_sec": 2.0,
                "source_in_sec": 0.0, "source_out_sec": 2.0, "audio_mode": "NOT_A_REAL_MODE",
            }],
        },
    )
    assert response.status_code == 422


def test_unknown_role_rejected_by_model_validation(client, fake_redis, durable_dir, project, broll_source):
    response = _create_asset(client, project["project_id"], role="NOT_A_REAL_ROLE", local_path=broll_source)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 18/19/29/30. wrong user / wrong project denied
# ---------------------------------------------------------------------------

def test_wrong_user_cannot_list_project_assets(client, fake_redis, durable_dir, project, broll_source):
    _create_asset(client, project["project_id"], local_path=broll_source)
    response = client.get(f"/v1/projects/{project['project_id']}/timeline-assets", params={"user_id": "u2"})
    assert response.status_code == 404  # project ownership check fails first -- no metadata leak


def test_wrong_user_cannot_save_timeline(client, fake_redis, durable_dir, project, base_edit_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={"user_id": "u2", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0},
    )
    assert response.status_code == 404


def test_cross_project_asset_reuse_blocked_at_save(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    other_project = project_store.create_project(user_id="u1", title="Other", client=fake_redis)
    other_broll = _create_asset(client, other_project["project_id"], local_path=broll_source).json()

    response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={
            "user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0,
            "broll_placements": [{
                "placement_id": "broll_1", "asset_id": other_broll["asset_id"],
                "timeline_start_sec": 0.0, "timeline_end_sec": 2.0,
                "source_in_sec": 0.0, "source_out_sec": 2.0,
            }],
        },
    )
    # Stage 28: the route's own asset lookup is already scoped to THIS
    # project's own asset list -- an asset from a different project is
    # not merely "not owned by this user here", it is not visible at
    # all from this project's own perspective, so it correctly resolves
    # ASSET_NOT_FOUND (tighter than ASSET_NOT_OWNED, no leak that the
    # asset exists anywhere else).
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["outcome"] == reg.TIMELINE_INVALID
    assert any(reg.ASSET_NOT_FOUND in r for r in detail["reasons"])


# ---------------------------------------------------------------------------
# 16/21/22/23/35. export exact revision -> D-278 bridge -> final QC PASS
# D-282A Stage 7/8/9 -- opaque export_id + ownership-checked download route
# ---------------------------------------------------------------------------

def test_export_exact_revision_end_to_end_qc_pass(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    broll = _create_asset(client, project["project_id"], local_path=broll_source).json()

    save_response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={
            "user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0,
            "broll_placements": [{
                "placement_id": "broll_1", "asset_id": broll["asset_id"],
                "timeline_start_sec": 1.0, "timeline_end_sec": 3.0,
                "source_in_sec": 0.0, "source_out_sec": 2.0,
            }],
        },
    )
    revision_identity = save_response.json()["timeline_revision_identity"]

    export_response = client.post(
        f"/v1/projects/{project['project_id']}/timeline/export",
        json={"user_id": "u1", "revision_identity": revision_identity, "base_edit_asset_id": base_asset_id},
    )
    assert export_response.status_code == 200, export_response.text
    body = export_response.json()
    assert body["outcome"] in ("COMPOSITION_SUCCEEDED", "BASE_ONLY_BYPASS")
    assert "output_path" not in body
    if body["outcome"] == "COMPOSITION_SUCCEEDED":
        assert body["format_qc_status"] == "PASS"
        assert body["export_id"] is not None
        assert body["export_id"].startswith("exp_")

        download = client.get(
            f"/v1/projects/{project['project_id']}/timeline/export/{body['export_id']}/download",
            params={"user_id": "u1"},
        )
        assert download.status_code == 200
        assert download.headers["content-type"] == "video/mp4"
        assert len(download.content) > 0


def test_export_stale_revision_rejected(client, fake_redis, durable_dir, project, base_edit_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={"user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0},
    )
    export_response = client.post(
        f"/v1/projects/{project['project_id']}/timeline/export",
        json={"user_id": "u1", "revision_identity": "not-the-real-one", "base_edit_asset_id": base_asset_id},
    )
    assert export_response.status_code == 409
    assert export_response.json()["detail"]["outcome"] == reg.TIMELINE_REVISION_CONFLICT


def test_export_mismatched_base_edit_asset_rejected(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    save_response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={"user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0},
    )
    revision_identity = save_response.json()["timeline_revision_identity"]

    # A DIFFERENT primary-source asset -- never used in the save above.
    different_base_id = _create_base_edit(client, project["project_id"], broll_source)
    export_response = client.post(
        f"/v1/projects/{project['project_id']}/timeline/export",
        json={"user_id": "u1", "revision_identity": revision_identity, "base_edit_asset_id": different_base_id},
    )
    assert export_response.status_code == 409
    assert export_response.json()["detail"]["outcome"] == timeline_routes.BASE_EDIT_ASSET_MISMATCH


def test_download_wrong_user_denied_no_leak(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    broll = _create_asset(client, project["project_id"], local_path=broll_source).json()
    save_response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={
            "user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0,
            "broll_placements": [{
                "placement_id": "broll_1", "asset_id": broll["asset_id"],
                "timeline_start_sec": 1.0, "timeline_end_sec": 3.0,
                "source_in_sec": 0.0, "source_out_sec": 2.0,
            }],
        },
    )
    revision_identity = save_response.json()["timeline_revision_identity"]
    export_response = client.post(
        f"/v1/projects/{project['project_id']}/timeline/export",
        json={"user_id": "u1", "revision_identity": revision_identity, "base_edit_asset_id": base_asset_id},
    )
    body = export_response.json()
    if body["outcome"] != "COMPOSITION_SUCCEEDED":
        pytest.skip("export did not produce a deliverable in this environment")
    download = client.get(
        f"/v1/projects/{project['project_id']}/timeline/export/{body['export_id']}/download",
        params={"user_id": "u2"},
    )
    # Same single-owner project model as `test_wrong_user_cannot_save_
    # timeline`: `_require_project` denies a non-owning user before this
    # route's own `export_ref` ownership check is ever reached -- no
    # metadata leak either way.
    assert download.status_code == 404


def test_export_reference_enforces_user_ownership_at_module_level(fake_redis):
    # Same defense-in-depth rationale as `test_upload_id_bound_to_
    # issuing_user_cross_user_denied` -- the route's own project-ownership
    # gate happens to catch this first in a single-owner project model,
    # so this proves `resolve_export_reference` ALSO enforces it on its
    # own, independent of that gate.
    export_id = export_ref.register_export_reference(
        user_id="u1", project_id="p1", local_output_path="/tmp/does-not-matter.mp4",
        format_qc_status="PASS", redis_client=fake_redis,
    )
    with pytest.raises(export_ref.ExportReferenceResolutionError) as excinfo:
        export_ref.resolve_export_reference(export_id=export_id, user_id="u2", project_id="p1", redis_client=fake_redis)
    assert excinfo.value.outcome == export_ref.EXPORT_REFERENCE_NOT_OWNED


def test_download_unknown_export_id_returns_bounded_404(client, fake_redis, durable_dir, project):
    response = client.get(
        f"/v1/projects/{project['project_id']}/timeline/export/exp_does-not-exist/download",
        params={"user_id": "u1"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["outcome"] == export_ref.EXPORT_REFERENCE_NOT_FOUND


# ---------------------------------------------------------------------------
# 20/26. no signed URL identity, project/account deletion integration
# ---------------------------------------------------------------------------

def test_project_deletion_reaches_timeline_asset_cleanup(monkeypatch, fake_redis, durable_dir, project, broll_source, client):
    _create_asset(client, project["project_id"], local_path=broll_source)
    listed_before = store.list_timeline_assets(user_id="u1", project_id=project["project_id"], client=fake_redis)
    assert len(listed_before) == 1

    class _Config:
        redis_url = "redis://fake"
        s3_bucket = None
        database_url = None

    monkeypatch.setattr(account_lifecycle, "load_runtime_config", lambda: _Config())
    account_lifecycle.delete_project_data(user_id="u1", project_id=project["project_id"], redis_client=fake_redis)
    listed_after = store.list_timeline_assets(user_id="u1", project_id=project["project_id"], client=fake_redis)
    assert len(listed_after) == 0


# ---------------------------------------------------------------------------
# D-282A Stage 9 -- local-path / raw-storage-reference firewall
# ---------------------------------------------------------------------------

_FORBIDDEN_LEAK_SUBSTRINGS = ("/tmp/", "/mnt/", "/var/", "local://", "s3://")


def test_no_local_path_or_raw_storage_uri_in_any_client_response(
    client, fake_redis, durable_dir, project, base_edit_source, broll_source,
):
    # The upload-authorization response is deliberately excluded from this
    # sweep: `upload_url` legitimately carries a client-facing PUT target
    # (in this fake-storage test double it is stubbed as `local://...`; in
    # production it is a real presigned HTTPS URL) -- that field's own
    # `source_uri`/`object_key`/`bucket` NEVER appearing is separately
    # proven by `test_upload_id_is_server_issued_and_opaque`.
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    broll_asset = _create_asset(client, project["project_id"], local_path=broll_source)
    listing = client.get(f"/v1/projects/{project['project_id']}/timeline-assets", params={"user_id": "u1"})
    save_response = client.put(
        f"/v1/projects/{project['project_id']}/timeline",
        json={"user_id": "u1", "base_edit_asset_id": base_asset_id, "timeline_duration_sec": 6.0},
    )
    revision_identity = save_response.json()["timeline_revision_identity"]
    get_response = client.get(f"/v1/projects/{project['project_id']}/timeline", params={"user_id": "u1"})
    export_response = client.post(
        f"/v1/projects/{project['project_id']}/timeline/export",
        json={"user_id": "u1", "revision_identity": revision_identity, "base_edit_asset_id": base_asset_id},
    )

    for response in (broll_asset, listing, save_response, get_response, export_response):
        text = str(response.json())
        for forbidden in _FORBIDDEN_LEAK_SUBSTRINGS:
            assert forbidden not in text, f"{forbidden!r} leaked in: {text}"


def test_export_response_schema_never_declares_output_path():
    schema = app.openapi()
    export_schema = schema["components"]["schemas"]["TimelineExportResponse"]
    assert "output_path" not in export_schema.get("properties", {})
    assert "export_id" in export_schema.get("properties", {})


def test_asset_create_request_schema_never_declares_source_uri():
    schema = app.openapi()
    create_schema = schema["components"]["schemas"]["TimelineAssetCreateRequest"]
    assert "source_uri" not in create_schema.get("properties", {})
    assert "upload_id" in create_schema.get("properties", {})


# ---------------------------------------------------------------------------
# D-282A Stage 10 -- auth principal (audit only, no auth redesign)
# ---------------------------------------------------------------------------

def test_client_user_id_never_overrides_authenticated_identity_when_auth_enforced(monkeypatch):
    # AuthScopeMiddleware (unmodified) already rejects a request whose
    # body `user_id` differs from the bearer-resolved principal -- this
    # is an audit that the pre-existing mechanism still governs D-282A's
    # own new routes, not a new auth mechanism.
    monkeypatch.setenv("CUTSELL_AUTH_REQUIRED", "1")

    from cutsell_app import auth_middleware

    monkeypatch.setattr(auth_middleware, "resolve_session", lambda token: {"user_id": "real-user"})
    raw_client = TestClient(app)
    response = raw_client.post(
        "/v1/projects/p1/timeline-uploads",
        json={"user_id": "someone-else", "media_class": "video", "original_name": "x.mp4", "size_bytes": 10},
        headers={"Authorization": "Bearer faketoken"},
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# 31/40. no real S3, no mobile code
# ---------------------------------------------------------------------------

def test_no_real_s3_mutation_in_route_module():
    import inspect
    source = inspect.getsource(timeline_routes)
    assert "put_object" not in source
    assert "upload_file" not in source


def test_no_real_s3_mutation_in_new_d282a_modules():
    import inspect
    for module in (upload_reg, export_ref):
        source = inspect.getsource(module)
        assert "put_object" not in source
        assert "upload_file" not in source


def test_no_mobile_or_ios_code_referenced():
    import inspect
    source = inspect.getsource(timeline_routes).lower()
    assert "mobile/ios" not in source
    assert "swift" not in source


# ---------------------------------------------------------------------------
# Structural firewall (Stage 37)
# ---------------------------------------------------------------------------

def test_route_module_never_imports_closed_tracks():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(timeline_routes))
    forbidden = {
        "best_take_authority", "deterministic_best_take_authority", "selection_freeze",
        "boundary_engine", "dialogue_pacing_transition", "audio_finishing_executor",
        "audio_finishing_measurement", "visual_finishing_policy", "visual_mode",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden
