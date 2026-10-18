"""Primary video at the end of a cut + real S3 storage for timeline assets.

A synthetic test cut (ffmpeg colour + tone, no real person or footage) is
registered as the project's PRIMARY_SOURCE through `timeline_asset_s3`, then
the same API calls the iPhone editor makes (voice-over upload, B-roll upload,
timeline save) run in S3 mode against an in-memory fake S3 and fake Redis.
No real S3, Redis or paid service is touched.
"""
from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cutsell_app import timeline_routes
from cutsell_app.main import app
from cutsell_worker import account_lifecycle
from cutsell_worker import project_store
from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker import timeline_asset_registry_store as store
from cutsell_worker import timeline_asset_s3 as s3_media
from cutsell_worker import timeline_export_reference as export_ref
from cutsell_worker import timeline_upload_registration as upload_reg
from cutsell_worker import uploads
from cutsell_worker import worker_job

BUCKET = "cutsell-test-bucket"
USER = "usr_probe"


# ---------------------------------------------------------------------------
# fakes
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
            getattr(self.redis, op[0])(*op[1:])
        self.ops = []
        return [True]


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.zsets = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):  # noqa: ARG002
        self.data[key] = value
        return True

    def delete(self, *keys):
        n = 0
        for key in keys:
            n += int(self.data.pop(key, None) is not None) + int(self.zsets.pop(key, None) is not None)
        return n

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    def zrevrange(self, key, start, end):
        ordered = sorted(self.zsets.get(key, {}).items(), key=lambda item: item[1], reverse=True)
        stop = None if end < 0 else end + 1
        return [(k.encode() if isinstance(k, str) else k) for k, _ in ordered[start:stop]]

    def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)

    def pipeline(self):
        return FakePipeline(self)


class FakeS3:
    """upload_file / download_file over a temp directory, keyed by bucket/key."""

    def __init__(self, root: Path):
        self.root = root
        self.uploads: list[str] = []
        self.downloads: list[str] = []

    def _path(self, bucket: str, key: str) -> Path:
        return self.root / bucket / key

    def put(self, bucket: str, key: str, local_path: str) -> None:
        target = self._path(bucket, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)

    def upload_file(self, filename, bucket, key):
        self.uploads.append(key)
        self.put(bucket, key, filename)

    def download_file(self, bucket, key, filename):
        self.downloads.append(key)
        source = self._path(bucket, key)
        if not source.exists():
            raise FileNotFoundError(key)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, filename)

    def has(self, uri: str) -> bool:
        bucket, key = uri[len("s3://"):].split("/", 1)
        return self._path(bucket, key).exists()


@pytest.fixture()
def fake_redis(monkeypatch):
    redis = FakeRedis()
    for module in (project_store, store, upload_reg, export_ref):
        monkeypatch.setattr(module, "_redis_client", lambda client=None, _r=redis: _r if client is None else client)
    return redis


@pytest.fixture()
def fake_s3(tmp_path, monkeypatch):
    s3 = FakeS3(tmp_path / "s3")
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("CUTSELL_UPLOAD_PREFIX", "cutsell/uploads/")
    monkeypatch.delenv("CUTSELL_TIMELINE_ASSET_DURABLE_DIR", raising=False)
    monkeypatch.delenv("CUTSELL_VOICE_OVER_UPLOAD_PREFIX", raising=False)
    monkeypatch.setattr(s3_media, "_client", lambda client=None: s3 if client is None else client)
    return s3


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True, capture_output=True)


@pytest.fixture(scope="module")
def synthetic_cut(tmp_path_factory):
    """The test cut's source video: a plain colour frame with a tone (no real footage)."""
    path = tmp_path_factory.mktemp("cut") / "probe.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x203060:s=720x1280:d=6:r=30", "-f", "lavfi", "-i",
            "sine=frequency=220:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-ar", "48000", "-ac", "2", "-shortest", str(path))
    return str(path)


@pytest.fixture(scope="module")
def broll_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("broll") / "broll.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=green:s=720x1280:d=2:r=30", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-ar", "48000", "-ac", "2", "-shortest", str(path))
    return str(path)


@pytest.fixture(scope="module")
def voice_take(tmp_path_factory):
    path = tmp_path_factory.mktemp("vo") / "voice.m4a"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=880:duration=1.5", "-c:a", "aac", "-ar", "48000", str(path))
    return str(path)


@pytest.fixture()
def project(fake_redis):
    return project_store.create_project(user_id=USER, title="Synthetic test cut", client=fake_redis)


def _uploaded_source(fake_s3: FakeS3, project_id: str, local_path: str) -> str:
    """Puts the test cut's source where a real upload lands (the project's upload scope)."""
    key = f"{uploads.scoped_upload_prefix(user_id=USER, project_id=project_id)}abc-probe.mp4"
    fake_s3.put(BUCKET, key, local_path)
    return f"s3://{BUCKET}/{key}"


def _cut_downloads(tmp_path: Path, synthetic_cut: str) -> str:
    """The worker's local copy of the source (what `download_source` produced)."""
    local = tmp_path / "job" / "000-src.mp4"
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(synthetic_cut, local)
    return str(local)


# ---------------------------------------------------------------------------
# primary video registration (worker side)
# ---------------------------------------------------------------------------

def test_primary_video_registered_from_the_uploaded_source_without_a_copy(
    tmp_path, fake_redis, fake_s3, project, synthetic_cut,
):
    pid = project["project_id"]
    source_uri = _uploaded_source(fake_s3, pid, synthetic_cut)
    local = _cut_downloads(tmp_path, synthetic_cut)

    result = s3_media.register_primary_video(
        user_id=USER, project_id=pid, source_uri=source_uri,
        original_local_path=local, cut_local_path=local, scratch_dir=str(tmp_path / "job"),
    )

    assert result["status"] == "registered"
    asset = store.get_timeline_asset(user_id=USER, project_id=pid, asset_id=result["asset_id"])
    assert asset.role == reg.TimelineAssetRole.PRIMARY_SOURCE
    assert asset.media_kind == reg.TimelineMediaKind.VIDEO
    assert asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
    assert asset.duration_sec == pytest.approx(6.0, abs=0.3)
    assert asset.storage_reference == source_uri  # the upload itself, not a second copy
    assert fake_s3.uploads == []
    # public-safe: no path or storage URI in what the job reports
    assert set(result) == {"status", "asset_id", "qualification_status"}


def test_primary_video_registration_is_idempotent(tmp_path, fake_redis, fake_s3, project, synthetic_cut):
    pid = project["project_id"]
    source_uri = _uploaded_source(fake_s3, pid, synthetic_cut)
    local = _cut_downloads(tmp_path, synthetic_cut)
    kwargs = dict(user_id=USER, project_id=pid, source_uri=source_uri, original_local_path=local,
                  cut_local_path=local, scratch_dir=str(tmp_path / "job"))
    first = s3_media.register_primary_video(**kwargs)
    second = s3_media.register_primary_video(**kwargs)
    assert second == {"status": "already_registered", "asset_id": first["asset_id"]}
    primaries = [a for a in store.list_timeline_assets(user_id=USER, project_id=pid)
                 if a.role == reg.TimelineAssetRole.PRIMARY_SOURCE]
    assert len(primaries) == 1


def test_normalized_source_is_stored_once_inside_the_project_scope(
    tmp_path, fake_redis, fake_s3, project, synthetic_cut,
):
    pid = project["project_id"]
    source_uri = _uploaded_source(fake_s3, pid, synthetic_cut)
    original = _cut_downloads(tmp_path, synthetic_cut)
    normalized = tmp_path / "job" / "000-src.normalized.mp4"
    shutil.copyfile(synthetic_cut, normalized)

    result = s3_media.register_primary_video(
        user_id=USER, project_id=pid, source_uri=source_uri,
        original_local_path=original, cut_local_path=str(normalized), scratch_dir=str(tmp_path / "job"),
    )

    assert result["status"] == "registered"
    asset = store.get_timeline_asset(user_id=USER, project_id=pid, asset_id=result["asset_id"])
    prefix = s3_media.timeline_media_prefix(user_id=USER, project_id=pid)
    assert asset.storage_reference == f"s3://{BUCKET}/{prefix}{result['asset_id']}.mp4"
    assert fake_s3.has(asset.storage_reference)
    # deleting the project removes it (timeline media lives in the upload prefix)
    assert any(f"{prefix}".startswith(p) for p in account_lifecycle._project_prefixes(user_id=USER, project_id=pid))


def test_registration_never_raises(tmp_path, fake_redis, fake_s3, project):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    result = s3_media.register_primary_video(
        user_id=USER, project_id=project["project_id"], source_uri=f"s3://{BUCKET}/x",
        original_local_path=str(broken), cut_local_path=str(broken), scratch_dir=str(tmp_path),
    )
    assert result["status"] in {"degraded", "not_ready"}
    assert not [a for a in store.list_timeline_assets(user_id=USER, project_id=project["project_id"])
                if a.qualification_status == reg.TimelineAssetQualificationStatus.READY]


def test_fetcher_only_reads_this_projects_objects(tmp_path, fake_s3):
    fetch = s3_media.s3_fetcher(user_id=USER, project_id="prj_a", scratch_dir=str(tmp_path))
    other = uploads.scoped_upload_prefix(user_id=USER, project_id="prj_b")
    with pytest.raises(RuntimeError):
        fetch(f"s3://{BUCKET}/{other}clip.mp4")
    with pytest.raises(RuntimeError):
        fetch(f"s3://another-bucket/{uploads.scoped_upload_prefix(user_id=USER, project_id='prj_a')}clip.mp4")
    mine = uploads.scoped_upload_prefix(user_id=USER, project_id="prj_a")
    with pytest.raises(RuntimeError):
        fetch(f"s3://{BUCKET}/{mine}../../escape.mp4")


def test_persister_reuse_is_limited_to_this_project(tmp_path, fake_s3, synthetic_cut):
    other = uploads.scoped_upload_prefix(user_id=USER, project_id="prj_b")
    persist = s3_media.s3_persister(
        user_id=USER, project_id="prj_a", reuse={synthetic_cut: f"s3://{BUCKET}/{other}x.mp4"},
    )
    with pytest.raises(RuntimeError):
        persist(synthetic_cut, "timeline-assets/a/b/tla_1.mp4")


def test_project_deletion_covers_voice_over_recordings():
    prefixes = account_lifecycle._project_prefixes(user_id=USER, project_id="prj_a")
    assert uploads.scoped_voice_over_upload_prefix(user_id=USER, project_id="prj_a") in prefixes


# ---------------------------------------------------------------------------
# cut job hook
# ---------------------------------------------------------------------------

def test_cut_job_registers_its_first_source(monkeypatch):
    seen = {}

    def fake_register(**kwargs):
        seen.update(kwargs)
        return {"status": "registered", "asset_id": "tla_x", "qualification_status": "READY"}

    monkeypatch.setattr(worker_job, "register_primary_video", fake_register)
    request = SimpleNamespace(
        user_id=USER, project_id="prj_a",
        sources=[SimpleNamespace(source_asset_id="s2", source_order=1, uri="s3://b/2"),
                 SimpleNamespace(source_asset_id="s1", source_order=0, uri="s3://b/1")],
    )
    out = worker_job._register_primary_video(
        request, {"s1": "/job/1.mp4", "s2": "/job/2.mp4"}, {"s1": "/job/1n.mp4", "s2": "/job/2.mp4"}, "/job",
    )
    assert out["status"] == "registered"
    assert seen["source_uri"] == "s3://b/1"
    assert seen["original_local_path"] == "/job/1.mp4"
    assert seen["cut_local_path"] == "/job/1n.mp4"


def test_cut_job_hook_never_fails_the_cut(monkeypatch):
    request = SimpleNamespace(user_id=USER, project_id="prj_a", sources=[])
    assert worker_job._register_primary_video(request, {}, {}, "/job")["status"] == "degraded"


def test_primary_video_is_registered_before_the_project_reads_draft_ready():
    source = inspect.getsource(worker_job.run_flow_b_job)
    # Build 23 (audio first): a voice-only first source waits for its video (follow-up job).
    hook = source.index('else _register_primary_video(request, downloaded_paths, local_paths, directory)')
    ready = source.index('state="draft_ready"')
    assert source.index("create_initial_draft(") < hook < ready
    assert source.index("downloaded_paths = dict(local_paths)") < source.index("resolve_sources_for_editorial_entry(video_paths")


# ---------------------------------------------------------------------------
# the editor's own calls, end to end, in S3 mode
# ---------------------------------------------------------------------------

def test_editor_can_place_voice_over_and_broll_after_a_cut(
    tmp_path, monkeypatch, fake_redis, fake_s3, project, synthetic_cut, broll_clip, voice_take,
):
    pid = project["project_id"]
    client = TestClient(app)

    # 1. end of the cut: the worker registers the primary video
    source_uri = _uploaded_source(fake_s3, pid, synthetic_cut)
    local = _cut_downloads(tmp_path, synthetic_cut)
    primary = s3_media.register_primary_video(
        user_id=USER, project_id=pid, source_uri=source_uri,
        original_local_path=local, cut_local_path=local, scratch_dir=str(tmp_path / "job"),
    )
    assert primary["status"] == "registered"

    # 2. the editor opens and lists the project's timeline assets
    listed = client.get(f"/v1/projects/{pid}/timeline-assets", params={"user_id": USER})
    assert listed.status_code == 200, listed.text
    roles = {(a["role"], a["qualification_status"]) for a in listed.json()["assets"]}
    assert ("PRIMARY_SOURCE", "READY") in roles

    # 3. the phone uploads a recording and a B-roll clip (presign -> PUT -> register)
    def presign_into_fake_s3(**kwargs):
        if kwargs["media_class"] == upload_reg.MEDIA_CLASS_VOICE_OVER:
            target = uploads.prepare_voice_over_upload_target(**{k: kwargs[k] for k in (
                "project_id", "user_id", "original_name", "content_type", "size_bytes")})
        else:
            target = uploads.prepare_upload_target(**{k: kwargs[k] for k in (
                "project_id", "user_id", "original_name", "content_type", "size_bytes")})
        local_file = voice_take if kwargs["media_class"] == upload_reg.MEDIA_CLASS_VOICE_OVER else broll_clip
        fake_s3.put(target["bucket"], target["object_key"], local_file)  # what the phone's PUT does
        return {"method": "POST", "upload_url": "https://example.invalid/upload", "fields": {},
                "source_uri": target["source_uri"], "object_key": target["object_key"],
                "content_type": target["content_type"], "max_bytes": target["size_bytes"],
                "expires_in": kwargs["expires_in"]}

    monkeypatch.setattr(upload_reg, "_default_presign_upload", presign_into_fake_s3)

    def add(media_class, name, content_type, size, role, kind):
        auth = client.post(f"/v1/projects/{pid}/timeline-uploads", json={
            "user_id": USER, "media_class": media_class, "original_name": name,
            "content_type": content_type, "size_bytes": size})
        assert auth.status_code == 200, auth.text
        assert "source_uri" not in auth.json()
        made = client.post(f"/v1/projects/{pid}/timeline-assets", json={
            "user_id": USER, "role": role, "media_kind": kind, "upload_id": auth.json()["upload_id"]})
        assert made.status_code == 200, made.text
        assert made.json()["qualification_status"] == "READY"
        return made.json()

    vo = add("voice_over", "voice-over.m4a", "audio/mp4", Path(voice_take).stat().st_size, "VOICE_OVER", "AUDIO")
    broll = add("video", "broll.mp4", "video/mp4", Path(broll_clip).stat().st_size, "SUPPLEMENTAL_BROLL", "VIDEO")

    # qualified media went to S3 inside the project scope; no scratch file survived
    prefix = s3_media.timeline_media_prefix(user_id=USER, project_id=pid)
    assert {k.rsplit("/", 1)[0] + "/" for k in fake_s3.uploads} == {prefix}
    assert len(fake_s3.uploads) == 2

    # 4. the editor saves the timeline with one voice-over and one B-roll placement
    saved = client.put(f"/v1/projects/{pid}/timeline", json={
        "user_id": USER, "base_edit_asset_id": primary["asset_id"], "timeline_duration_sec": 6.0,
        "broll_placements": [{"placement_id": "b1", "asset_id": broll["asset_id"], "timeline_start_sec": 1.0,
                              "timeline_end_sec": 2.5, "source_in_sec": 0.0, "source_out_sec": 1.5}],
        "voice_over_placements": [{"placement_id": "v1", "asset_id": vo["asset_id"], "timeline_start_sec": 3.0,
                                   "timeline_end_sec": 4.5, "source_in_sec": 0.0, "source_out_sec": 1.5}],
    })
    assert saved.status_code == 200, saved.text
    assert saved.json()["outcome"] == reg.TIMELINE_SAVE_SUCCEEDED

    # 5. reopening the editor shows both placements
    timeline = client.get(f"/v1/projects/{pid}/timeline", params={"user_id": USER})
    assert timeline.status_code == 200, timeline.text
    body = timeline.json()
    assert [p["asset_id"] for p in body["broll_placements"]] == [broll["asset_id"]]
    assert [p["asset_id"] for p in body["voice_over_placements"]] == [vo["asset_id"]]


def test_ingest_without_any_storage_is_still_unavailable(monkeypatch, fake_redis, project):
    monkeypatch.delenv("CUTSELL_TIMELINE_ASSET_DURABLE_DIR", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setattr(upload_reg, "_default_presign_upload", lambda **k: {
        "method": "POST", "upload_url": "x", "fields": {}, "source_uri": "local:///nope",
        "object_key": "k", "content_type": "video/mp4", "max_bytes": 1, "expires_in": 900})
    client = TestClient(app)
    pid = project["project_id"]
    auth = client.post(f"/v1/projects/{pid}/timeline-uploads", json={
        "user_id": USER, "media_class": "video", "original_name": "a.mp4", "content_type": "video/mp4",
        "size_bytes": 10})
    made = client.post(f"/v1/projects/{pid}/timeline-assets", json={
        "user_id": USER, "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": auth.json()["upload_id"]})
    assert made.status_code == 503


# ---------------------------------------------------------------------------
# review fixes: export in S3 mode, bounded errors, uploads not burned
# ---------------------------------------------------------------------------

def _registered_primary(tmp_path, fake_s3, pid, synthetic_cut):
    source_uri = _uploaded_source(fake_s3, pid, synthetic_cut)
    local = _cut_downloads(tmp_path, synthetic_cut)
    return s3_media.register_primary_video(
        user_id=USER, project_id=pid, source_uri=source_uri,
        original_local_path=local, cut_local_path=local, scratch_dir=str(tmp_path / "job"),
    )


def test_base_only_export_in_s3_mode_can_still_be_downloaded(
    tmp_path, monkeypatch, fake_redis, fake_s3, project, synthetic_cut,
):
    (tmp_path / "srvtmp").mkdir()
    monkeypatch.setattr(timeline_routes.tempfile, "gettempdir", lambda: str(tmp_path / "srvtmp"))
    pid = project["project_id"]
    primary = _registered_primary(tmp_path, fake_s3, pid, synthetic_cut)
    client = TestClient(app)
    saved = client.put(f"/v1/projects/{pid}/timeline", json={
        "user_id": USER, "base_edit_asset_id": primary["asset_id"], "timeline_duration_sec": 6.0})
    assert saved.status_code == 200, saved.text
    exported = client.post(f"/v1/projects/{pid}/timeline/export", json={
        "user_id": USER, "revision_identity": saved.json()["timeline_revision_identity"],
        "base_edit_asset_id": primary["asset_id"]})
    assert exported.status_code == 200, exported.text
    download = client.get(f"/v1/projects/{pid}/timeline/export/{exported.json()['export_id']}/download",
                          params={"user_id": USER})
    assert download.status_code == 200
    assert len(download.content) == Path(synthetic_cut).stat().st_size


def test_missing_export_file_is_a_bounded_404(tmp_path, monkeypatch, fake_redis, fake_s3, project):
    pid = project["project_id"]
    export_id = export_ref.register_export_reference(
        user_id=USER, project_id=pid, local_output_path=str(tmp_path / "gone.mp4"), format_qc_status=None)
    response = TestClient(app).get(f"/v1/projects/{pid}/timeline/export/{export_id}/download",
                                   params={"user_id": USER})
    assert response.status_code == 404


def test_storage_failure_is_a_bounded_503(tmp_path, monkeypatch, fake_redis, fake_s3, project):
    pid = project["project_id"]
    key = f"{uploads.scoped_upload_prefix(user_id=USER, project_id=pid)}never-uploaded.mp4"
    monkeypatch.setattr(upload_reg, "_default_presign_upload", lambda **k: {
        "method": "POST", "upload_url": "x", "fields": {}, "source_uri": f"s3://{BUCKET}/{key}",
        "object_key": key, "content_type": "video/mp4", "max_bytes": 10, "expires_in": 900})
    client = TestClient(app, raise_server_exceptions=False)
    auth = client.post(f"/v1/projects/{pid}/timeline-uploads", json={
        "user_id": USER, "media_class": "video", "original_name": "a.mp4", "content_type": "video/mp4",
        "size_bytes": 10})
    made = client.post(f"/v1/projects/{pid}/timeline-assets", json={
        "user_id": USER, "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": auth.json()["upload_id"]})
    assert made.status_code == 503
    assert "FileNotFoundError" in made.text


def test_unconfigured_storage_does_not_consume_the_upload(monkeypatch, fake_redis, fake_s3, project):
    pid = project["project_id"]
    monkeypatch.setattr(upload_reg, "_default_presign_upload", lambda **k: {
        "method": "POST", "upload_url": "x", "fields": {}, "source_uri": "local:///nope",
        "object_key": "k", "content_type": "video/mp4", "max_bytes": 10, "expires_in": 900})
    client = TestClient(app)
    auth = client.post(f"/v1/projects/{pid}/timeline-uploads", json={
        "user_id": USER, "media_class": "video", "original_name": "a.mp4", "content_type": "video/mp4",
        "size_bytes": 10})
    upload_id = auth.json()["upload_id"]
    monkeypatch.delenv("S3_BUCKET")
    body = {"user_id": USER, "role": "SUPPLEMENTAL_BROLL", "media_kind": "VIDEO", "upload_id": upload_id}
    assert client.post(f"/v1/projects/{pid}/timeline-assets", json=body).status_code == 503
    # still usable afterwards (not UPLOAD_ALREADY_CONSUMED)
    assert upload_reg.resolve_and_consume_timeline_upload(
        upload_id=upload_id, user_id=USER, project_id=pid, expected_media_class="video") == "local:///nope"
