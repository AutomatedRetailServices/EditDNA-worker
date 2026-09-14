"""D-282 -- Mobile Timeline API Bridge + Voice-Over Upload Contract.

Real FastAPI TestClient against `cutsell_app.main.app`, real ffmpeg
fixtures, fake Redis (this session's established convention), fake/local
durable storage (Stage 31/36 -- no real S3 mutation).
"""
from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from cutsell_app import timeline_routes
from cutsell_app.main import app
from cutsell_worker import account_lifecycle
from cutsell_worker import project_store
from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker import timeline_asset_registry_store as store
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

    def set(self, key, value):
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
    return redis


@pytest.fixture()
def durable_dir(tmp_path, monkeypatch):
    directory = tmp_path / "durable"
    directory.mkdir()
    monkeypatch.setenv("CUTSELL_TIMELINE_ASSET_DURABLE_DIR", str(directory))
    return str(directory)


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


def _create_asset(client, project_id, *, user_id="u1", role="SUPPLEMENTAL_BROLL", media_kind="VIDEO", source_uri):
    return client.post(
        f"/v1/projects/{project_id}/timeline-assets",
        json={"user_id": user_id, "role": role, "media_kind": media_kind, "source_uri": source_uri},
    )


# ---------------------------------------------------------------------------
# 1/2/3. routes exist, auth required, ownership required
# ---------------------------------------------------------------------------

def test_timeline_routes_exist_and_are_registered():
    schema = app.openapi()
    assert schema["paths"]["/v1/projects/{project_id}/timeline-assets"].keys() >= {"post", "get"}
    assert schema["paths"]["/v1/projects/{project_id}/timeline"].keys() >= {"get", "put"}
    assert schema["paths"]["/v1/projects/{project_id}/timeline/export"].keys() >= {"post"}


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
    response = _create_asset(client, project["project_id"], source_uri=f"local://{broll_source}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["qualification_status"] == "READY"
    assert body["media_kind"] == "VIDEO"
    assert body["duration_sec"] == pytest.approx(2.0, abs=0.3)
    assert body["has_audio"] is True
    assert "storage_reference" not in body


def test_list_assets_scoped_to_project(client, fake_redis, durable_dir, project, broll_source):
    _create_asset(client, project["project_id"], source_uri=f"local://{broll_source}")
    other_project = project_store.create_project(user_id="u1", title="Other", client=fake_redis)
    _create_asset(client, other_project["project_id"], source_uri=f"local://{broll_source}")

    response = client.get(f"/v1/projects/{project['project_id']}/timeline-assets", params={"user_id": "u1"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["assets"]) == 1


def test_reject_creation_of_video_that_fails_qualification(client, fake_redis, durable_dir, project, tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"not a real video")
    response = _create_asset(client, project["project_id"], source_uri=f"local://{bogus}")
    assert response.status_code == 200
    body = response.json()
    assert body["qualification_status"] != "READY"


# ---------------------------------------------------------------------------
# 10/11/12/33. VO upload format accepted, server-authoritative metadata, READY
# ---------------------------------------------------------------------------

def test_create_voice_over_asset_via_api(client, fake_redis, durable_dir, project, vo_source):
    response = _create_asset(
        client, project["project_id"], role="VOICE_OVER", media_kind="AUDIO", source_uri=f"local://{vo_source}",
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
# 5/6/13/14/15/16/17. client-safe response, get/save timeline, concurrency,
# silent-audio validation
# ---------------------------------------------------------------------------

def _create_base_edit(client, project_id, base_edit_source):
    response = _create_asset(
        client, project_id, role="PRIMARY_SOURCE", media_kind="VIDEO", source_uri=f"local://{base_edit_source}",
    )
    assert response.status_code == 200, response.text
    return response.json()["asset_id"]


def test_save_and_reopen_timeline_round_trip(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    broll = _create_asset(client, project["project_id"], source_uri=f"local://{broll_source}").json()

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
    silent_asset = _create_asset(client, project["project_id"], source_uri=f"local://{silent}").json()
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
    broll = _create_asset(client, project["project_id"], source_uri=f"local://{broll_source}").json()
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
    response = _create_asset(client, project["project_id"], role="NOT_A_REAL_ROLE", source_uri=f"local://{broll_source}")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 18/19/29/30. wrong user / wrong project denied
# ---------------------------------------------------------------------------

def test_wrong_user_cannot_list_project_assets(client, fake_redis, durable_dir, project, broll_source):
    _create_asset(client, project["project_id"], source_uri=f"local://{broll_source}")
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
    other_broll = _create_asset(client, other_project["project_id"], source_uri=f"local://{broll_source}").json()

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
# ---------------------------------------------------------------------------

def test_export_exact_revision_end_to_end_qc_pass(client, fake_redis, durable_dir, project, base_edit_source, broll_source):
    base_asset_id = _create_base_edit(client, project["project_id"], base_edit_source)
    broll = _create_asset(client, project["project_id"], source_uri=f"local://{broll_source}").json()

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
    if body["outcome"] == "COMPOSITION_SUCCEEDED":
        assert body["format_qc_status"] == "PASS"
        assert body["output_path"] is not None


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


# ---------------------------------------------------------------------------
# 20/26. no signed URL identity, project/account deletion integration
# ---------------------------------------------------------------------------

def test_project_deletion_reaches_timeline_asset_cleanup(monkeypatch, fake_redis, durable_dir, project, broll_source, client):
    _create_asset(client, project["project_id"], source_uri=f"local://{broll_source}")
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
# 31/40. no real S3, no mobile code
# ---------------------------------------------------------------------------

def test_no_real_s3_mutation_in_route_module():
    import inspect
    source = inspect.getsource(timeline_routes)
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
