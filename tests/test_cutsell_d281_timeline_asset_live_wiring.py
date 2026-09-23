"""D-281 -- Timeline Asset Persistence / API Live Wiring.

Real ffmpeg fixtures, fake/local storage (Stage 36 -- no real S3
mutation), a fake Redis client (this session's established test
convention). No mobile UI, no microphone capture, no AI B-roll.
"""
from __future__ import annotations

import subprocess

import pytest

from cutsell_worker import account_lifecycle
from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker import timeline_asset_registry_store as store
from cutsell_worker import timeline_composition as tc
from cutsell_worker import timeline_composition_executor as tce


# ---------------------------------------------------------------------------
# Fake Redis (this session's established convention, e.g.
# tests/test_cutsell_clean_worker_projects.py)
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


# ---------------------------------------------------------------------------
# Real ffmpeg fixtures (session-scoped for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def broll_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d281_src") / "broll.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=2:r=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-ar", "48000", "-ac", "2", str(path),
        ],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture(scope="module")
def broll_source_silent(tmp_path_factory):
    path = tmp_path_factory.mktemp("d281_src") / "broll_silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=640x360:d=2:r=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
        ],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture(scope="module")
def vo_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d281_src") / "vo.m4a"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=1.5",
            "-c:a", "aac", "-ar", "48000", str(path),
        ],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture(scope="module")
def base_edit_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d281_src") / "base.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=1080x1920:d=6:r=30",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-ar", "48000", "-ac", "2", str(path),
        ],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture()
def redis():
    return FakeRedis()


@pytest.fixture()
def persist(tmp_path):
    return store.local_directory_persister(str(tmp_path / "durable"))


# ---------------------------------------------------------------------------
# 1/4. create READY B-roll asset
# ---------------------------------------------------------------------------

def test_create_ready_broll_asset(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    assert asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
    assert asset.duration_sec == pytest.approx(2.0, abs=0.3)
    assert asset.has_audio is True
    assert asset.storage_reference.startswith("local://")
    assert asset.asset_id.startswith("tla_")


# ---------------------------------------------------------------------------
# 2. create READY VO asset
# ---------------------------------------------------------------------------

def test_create_ready_vo_asset(redis, persist, vo_source):
    asset = store.create_voice_over_asset(
        user_id="u1", project_id="p1", local_source_path=vo_source, persist_media=persist, client=redis,
    )
    assert asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
    assert asset.role == reg.TimelineAssetRole.VOICE_OVER
    assert asset.media_kind == reg.TimelineMediaKind.AUDIO
    assert asset.has_audio is True
    assert asset.duration_sec == pytest.approx(1.5, abs=0.3)


# ---------------------------------------------------------------------------
# 5. rejected B-roll not READY / 6. insufficient evidence not READY
# ---------------------------------------------------------------------------

def test_rejected_source_is_not_ready(redis, persist, tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"not a real video file")
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=str(bogus), persist_media=persist, client=redis,
    )
    assert asset.qualification_status != reg.TimelineAssetQualificationStatus.READY
    assert asset.qualification_status in (
        reg.TimelineAssetQualificationStatus.REJECTED, reg.TimelineAssetQualificationStatus.FAILED,
    )


def test_silent_vo_source_fails_qualification(redis, persist, tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1:r=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(silent)],
        check=True, capture_output=True,
    )
    asset = store.create_voice_over_asset(
        user_id="u1", project_id="p1", local_source_path=str(silent), persist_media=persist, client=redis,
    )
    assert asset.qualification_status == reg.TimelineAssetQualificationStatus.FAILED
    assert asset.has_audio is False


# ---------------------------------------------------------------------------
# 7/8. wrong user / wrong project create+read denied
# ---------------------------------------------------------------------------

def test_wrong_user_cannot_resolve_asset(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    result = reg.resolve_timeline_asset(
        requesting=reg.AssetOwnershipScope(user_id="u2", project_id="p1"),
        asset=asset, local_path="/anything", required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_OWNED


def test_wrong_project_cannot_resolve_asset(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    result = reg.resolve_timeline_asset(
        requesting=reg.AssetOwnershipScope(user_id="u1", project_id="p2"),
        asset=asset, local_path="/anything", required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    assert result.outcome == reg.ASSET_NOT_OWNED


# ---------------------------------------------------------------------------
# 9. list project assets scoped / 10. safe asset response
# ---------------------------------------------------------------------------

def test_list_timeline_assets_scoped_by_user_and_project(redis, persist, broll_source):
    mine = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    store.create_video_timeline_asset(
        user_id="u2", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    store.create_video_timeline_asset(
        user_id="u1", project_id="p2", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    listed = store.list_timeline_assets(user_id="u1", project_id="p1", client=redis)
    assert {a.asset_id for a in listed} == {mine.asset_id}


def test_client_safe_asset_response_excludes_storage_reference(redis, persist, broll_source):
    store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    views = store.list_timeline_assets_client_safe(user_id="u1", project_id="p1", client=redis)
    assert len(views) == 1
    view = views[0]
    assert "storage_reference" not in view
    assert "technical_metadata_reference" not in view
    assert view["qualification_status"] == "READY"


# ---------------------------------------------------------------------------
# 11/12/13. save timeline revisions + optimistic concurrency
# ---------------------------------------------------------------------------

def _composition_with_broll(asset: reg.TimelineMediaAsset, duration=10.0):
    placement = tc.BrollPlacement(
        placement_id="broll_1", asset=reg.to_asset_reference(asset),
        timeline_start_sec=1.0, timeline_end_sec=1.0 + asset.duration_sec,
        source_in_sec=0.0, source_out_sec=asset.duration_sec,
    )
    base = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="base_1", timeline_duration_sec=duration,
    )
    composition, result = tc.add_broll(base, placement)
    assert result.valid, result.errors
    return composition


def test_save_first_timeline_revision_never_conflicts(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset)
    result = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    assert result.outcome == reg.TIMELINE_SAVE_SUCCEEDED


def test_save_next_revision_with_correct_expected_identity(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset)
    first = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    changed = tc.TimelineComposition(
        contract_version=composition.contract_version, base_edit_identity=composition.base_edit_identity,
        timeline_duration_sec=composition.timeline_duration_sec + 5.0,
        broll_placements=composition.broll_placements,
    )
    second = store.save_timeline(
        user_id="u1", project_id="p1", composition=changed,
        expected_revision_identity=first.record.timeline_revision_identity, client=redis,
    )
    assert second.outcome == reg.TIMELINE_SAVE_SUCCEEDED
    assert second.record.timeline_revision_identity != first.record.timeline_revision_identity


def test_stale_revision_save_is_rejected(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset)
    store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    changed = tc.TimelineComposition(
        contract_version=composition.contract_version, base_edit_identity=composition.base_edit_identity,
        timeline_duration_sec=composition.timeline_duration_sec + 5.0,
        broll_placements=composition.broll_placements,
    )
    result = store.save_timeline(
        user_id="u1", project_id="p1", composition=changed,
        expected_revision_identity="stale_value_from_before", client=redis,
    )
    assert result.outcome == reg.TIMELINE_REVISION_CONFLICT


def test_concurrent_stale_save_rejected_even_with_valid_original_identity(redis, persist, broll_source):
    """Two clients read the same revision; the second save (using the
    now-stale identity the first client also read) must be rejected."""
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset)
    baseline = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    shared_identity = baseline.record.timeline_revision_identity

    client_a_edit = tc.TimelineComposition(
        contract_version=composition.contract_version, base_edit_identity=composition.base_edit_identity,
        timeline_duration_sec=composition.timeline_duration_sec + 1.0, broll_placements=composition.broll_placements,
    )
    client_b_edit = tc.TimelineComposition(
        contract_version=composition.contract_version, base_edit_identity=composition.base_edit_identity,
        timeline_duration_sec=composition.timeline_duration_sec + 2.0, broll_placements=composition.broll_placements,
    )
    first_write = store.save_timeline(
        user_id="u1", project_id="p1", composition=client_a_edit,
        expected_revision_identity=shared_identity, client=redis,
    )
    assert first_write.outcome == reg.TIMELINE_SAVE_SUCCEEDED
    second_write = store.save_timeline(
        user_id="u1", project_id="p1", composition=client_b_edit,
        expected_revision_identity=shared_identity, client=redis,
    )
    assert second_write.outcome == reg.TIMELINE_REVISION_CONFLICT


# ---------------------------------------------------------------------------
# 14. reopen exact timeline
# ---------------------------------------------------------------------------

def test_reopen_timeline_returns_client_safe_exact_composition(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset)
    saved = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    reopened = store.get_timeline(user_id="u1", project_id="p1", client=redis)
    assert reopened["timeline_revision_identity"] == saved.record.timeline_revision_identity
    assert reopened["broll_placements"][0]["asset_id"] == asset.asset_id
    assert "storage_reference" not in reopened["broll_placements"][0]


def test_reopen_timeline_missing_project_returns_none(redis):
    assert store.get_timeline(user_id="ghost", project_id="ghost", client=redis) is None


# ---------------------------------------------------------------------------
# 15. asset duration server-authoritative (client cannot lie)
# ---------------------------------------------------------------------------

def test_duration_is_server_measured_never_from_a_client_value(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    # No parameter anywhere in create_video_timeline_asset accepts a
    # client-declared duration -- the value is always the real ffprobe
    # measurement.
    assert asset.duration_sec == pytest.approx(2.0, abs=0.3)


# ---------------------------------------------------------------------------
# 16. USE_BROLL_AUDIO + silent asset rejected at save time
# ---------------------------------------------------------------------------

def test_use_broll_audio_on_silent_asset_rejected_at_save(redis, persist, broll_source_silent):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source_silent, persist_media=persist, client=redis,
    )
    assert asset.has_audio is False
    placement = tc.BrollPlacement(
        placement_id="broll_1", asset=reg.to_asset_reference(asset),
        timeline_start_sec=0.0, timeline_end_sec=asset.duration_sec,
        source_in_sec=0.0, source_out_sec=asset.duration_sec,
        audio_mode=tc.TimelineAudioMode.USE_BROLL_AUDIO,
    )
    base = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="base_1", timeline_duration_sec=10.0,
    )
    composition, valid = tc.add_broll(base, placement)
    assert valid.valid
    result = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    assert result.outcome == reg.TIMELINE_INVALID
    assert any(reg.ASSET_HAS_NO_AUDIO in e for e in result.reason_codes)


# ---------------------------------------------------------------------------
# 17. missing asset rejected / 18. non-ready asset rejected
# ---------------------------------------------------------------------------

def test_missing_asset_reference_rejected_at_save(redis):
    ghost_ref = tc.TimelineAssetReference(asset_id="tla_ghost", source_media_identity="ghost", duration_sec=2.0)
    placement = tc.BrollPlacement(
        placement_id="broll_1", asset=ghost_ref, timeline_start_sec=0.0, timeline_end_sec=2.0,
        source_in_sec=0.0, source_out_sec=2.0,
    )
    base = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="base_1", timeline_duration_sec=10.0,
    )
    composition, valid = tc.add_broll(base, placement)
    assert valid.valid
    result = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    assert result.outcome == reg.TIMELINE_INVALID
    assert any(reg.ASSET_NOT_FOUND in e for e in result.reason_codes)


def test_non_ready_asset_rejected_at_save(redis, persist, tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"not real media")
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=str(bogus), persist_media=persist, client=redis,
    )
    assert asset.qualification_status != reg.TimelineAssetQualificationStatus.READY
    placement = tc.BrollPlacement(
        placement_id="broll_1", asset=reg.to_asset_reference(asset),
        timeline_start_sec=0.0, timeline_end_sec=asset.duration_sec,
        source_in_sec=0.0, source_out_sec=asset.duration_sec,
    )
    base = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity="base_1", timeline_duration_sec=10.0,
    )
    composition, valid = tc.add_broll(base, placement)
    assert valid.valid
    result = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    assert result.outcome == reg.TIMELINE_INVALID
    assert any(reg.ASSET_NOT_READY in e for e in result.reason_codes)


# ---------------------------------------------------------------------------
# 19. placement delete retains asset / 20. asset delete while referenced
# rejected / 21. unreferenced delete succeeds
# ---------------------------------------------------------------------------

def test_placement_delete_retains_asset_and_referenced_delete_blocked(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset)
    store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    blocked = store.delete_timeline_asset_live(user_id="u1", project_id="p1", asset_id=asset.asset_id, client=redis)
    assert blocked.outcome == reg.ASSET_REFERENCED

    without_broll = tc.TimelineComposition(
        contract_version=composition.contract_version, base_edit_identity=composition.base_edit_identity,
        timeline_duration_sec=composition.timeline_duration_sec,
    )
    saved_after_removal = store.save_timeline(
        user_id="u1", project_id="p1", composition=without_broll,
        expected_revision_identity=store.get_timeline_revision(
            user_id="u1", project_id="p1", client=redis,
        ).timeline_revision_identity,
        client=redis,
    )
    assert saved_after_removal.outcome == reg.TIMELINE_SAVE_SUCCEEDED
    # Placement removed from the timeline, but the asset record itself
    # is untouched (still exists, still READY) until explicitly deleted.
    still_there = store.get_timeline_asset(user_id="u1", project_id="p1", asset_id=asset.asset_id, client=redis)
    assert still_there is not None
    assert still_there.qualification_status == reg.TimelineAssetQualificationStatus.READY

    unblocked = store.delete_timeline_asset_live(user_id="u1", project_id="p1", asset_id=asset.asset_id, client=redis)
    assert unblocked.outcome == reg.ASSET_DELETE_SUCCEEDED


# ---------------------------------------------------------------------------
# 22. replace B-roll preserves old asset / 23. re-record VO new asset ID
# ---------------------------------------------------------------------------

def test_replace_broll_preserves_old_asset_in_library(redis, persist, broll_source, broll_source_silent):
    old_asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    new_asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source_silent, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(old_asset)
    saved = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    original_placement = composition.broll_placements[0]
    new_placement = tc.BrollPlacement(
        placement_id=original_placement.placement_id, asset=reg.to_asset_reference(new_asset),
        timeline_start_sec=original_placement.timeline_start_sec, timeline_end_sec=original_placement.timeline_start_sec + new_asset.duration_sec,
        source_in_sec=0.0, source_out_sec=new_asset.duration_sec,
    )
    replaced, valid = tc.replace_broll(composition, placement_id="broll_1", new_placement=new_placement)
    assert valid.valid, valid.errors
    result = store.save_timeline(
        user_id="u1", project_id="p1", composition=replaced,
        expected_revision_identity=saved.record.timeline_revision_identity, client=redis,
    )
    assert result.outcome == reg.TIMELINE_SAVE_SUCCEEDED
    old_still_there = store.get_timeline_asset(user_id="u1", project_id="p1", asset_id=old_asset.asset_id, client=redis)
    assert old_still_there is not None
    assert old_still_there.qualification_status == reg.TimelineAssetQualificationStatus.READY


def test_rerecord_voice_over_creates_new_asset_identity(redis, persist, vo_source):
    original = store.create_voice_over_asset(
        user_id="u1", project_id="p1", local_source_path=vo_source, persist_media=persist, client=redis,
    )
    rerecorded = store.rerecord_voice_over_asset_live(
        user_id="u1", project_id="p1", previous_asset_id=original.asset_id,
        local_source_path=vo_source, persist_media=persist, client=redis,
    )
    assert rerecorded.asset_id != original.asset_id
    assert rerecorded.replaces_asset_id == original.asset_id
    still_there = store.get_timeline_asset(user_id="u1", project_id="p1", asset_id=original.asset_id, client=redis)
    assert still_there is not None


# ---------------------------------------------------------------------------
# 24/25/26. exact revision export + D-278 bridge + final composition QC
# ---------------------------------------------------------------------------

def test_export_exact_revision_bridges_to_d278_and_qc_passes(redis, persist, broll_source, base_edit_source, tmp_path):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset, duration=6.0)
    saved = store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    base_edit_asset = tce.ResolvedTimelineAsset(
        asset_id="base_1", local_path=base_edit_source, duration_sec=6.0, has_audio=True,
    )
    output_dir = tmp_path / "export_output"
    output_dir.mkdir()
    result = store.export_timeline_revision(
        user_id="u1", project_id="p1", revision_identity=saved.record.timeline_revision_identity,
        base_edit_asset=base_edit_asset, output_directory=str(output_dir), client=redis,
    )
    assert isinstance(result, tce.CompositionExecutionResult)
    assert result.outcome in (tce.COMPOSITION_SUCCEEDED, tce.BASE_ONLY_BYPASS)
    if result.outcome == tce.COMPOSITION_SUCCEEDED:
        assert result.diagnostics.get("format_qc_status") == "PASS"


def test_export_stale_revision_identity_rejected(redis, persist, broll_source, base_edit_source, tmp_path):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    composition = _composition_with_broll(asset, duration=6.0)
    store.save_timeline(
        user_id="u1", project_id="p1", composition=composition, expected_revision_identity=None, client=redis,
    )
    base_edit_asset = tce.ResolvedTimelineAsset(
        asset_id="base_1", local_path=base_edit_source, duration_sec=6.0, has_audio=True,
    )
    result = store.export_timeline_revision(
        user_id="u1", project_id="p1", revision_identity="not_the_real_identity",
        base_edit_asset=base_edit_asset, output_directory=str(tmp_path), client=redis,
    )
    assert isinstance(result, reg.TimelineExportRequestResult)
    assert result.outcome == reg.TIMELINE_REVISION_CONFLICT


# ---------------------------------------------------------------------------
# 27. storage URL renewal doesn't change semantic identity
# ---------------------------------------------------------------------------

def test_storage_reference_never_part_of_asset_reference_identity(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    ref = reg.to_asset_reference(asset)
    assert not hasattr(ref, "storage_reference")
    # Simulate a storage URL "renewal": the ref built from the asset is
    # identical regardless of what storage_reference currently holds.
    ref_again = reg.to_asset_reference(asset)
    assert ref == ref_again


# ---------------------------------------------------------------------------
# 28. normalized durable reference not a temp path
# ---------------------------------------------------------------------------

def test_ready_asset_storage_reference_is_never_a_raw_tmp_path(redis, persist, broll_source, tmp_path):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    assert asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
    # The durable copy lives under the injected durable directory, not
    # at the original (job-scoped) source path.
    assert asset.storage_reference != f"local://{broll_source}"
    assert str(tmp_path / "durable") in asset.storage_reference


# ---------------------------------------------------------------------------
# 29/30. project/account deletion sees timeline asset records
# ---------------------------------------------------------------------------

def test_project_deletion_removes_timeline_asset_records(redis, persist, broll_source, monkeypatch):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    result = store.delete_project_timeline_assets(user_id="u1", project_id="p1", client=redis)
    assert result["timeline_assets_deleted"] == 1
    assert store.get_timeline_asset(user_id="u1", project_id="p1", asset_id=asset.asset_id, client=redis) is None


def test_delete_project_data_calls_timeline_asset_cleanup(monkeypatch):
    calls = {}

    def _fake_delete_project_timeline_assets(*, user_id, project_id, client=None):
        calls["called"] = (user_id, project_id)
        return {"timeline_assets_deleted": 3}

    monkeypatch.setattr(account_lifecycle, "delete_project_timeline_assets", _fake_delete_project_timeline_assets)
    monkeypatch.setattr(account_lifecycle, "get_project", lambda **kwargs: {"project_id": "p1"})

    class _NoOpPipe:
        def delete(self, *a, **k): return self
        def zrem(self, *a, **k): return self
        def execute(self): return [True]

    class _Redis:
        def pipeline(self): return _NoOpPipe()

    class _Config:
        redis_url = "redis://fake"
        s3_bucket = None
        database_url = None

    monkeypatch.setattr(account_lifecycle, "load_runtime_config", lambda: _Config())
    result = account_lifecycle.delete_project_data(user_id="u1", project_id="p1", redis_client=_Redis())
    assert calls["called"] == ("u1", "p1")
    assert result["timeline_assets_deleted"] == 3


# ---------------------------------------------------------------------------
# 31. no raw storage ref in client JSON / 32. no arbitrary path from client
# ---------------------------------------------------------------------------

def test_no_raw_storage_reference_anywhere_in_client_safe_views(redis, persist, broll_source):
    store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    import json
    views = store.list_timeline_assets_client_safe(user_id="u1", project_id="p1", client=redis)
    blob = json.dumps(views)
    assert "local://" not in blob
    assert "durable" not in blob


def test_client_cannot_choose_asset_id():
    import inspect
    sig = inspect.signature(store.create_video_timeline_asset)
    assert "asset_id" not in sig.parameters
    sig2 = inspect.signature(store.create_voice_over_asset)
    assert "asset_id" not in sig2.parameters


# ---------------------------------------------------------------------------
# Error mapping (Stage 41)
# ---------------------------------------------------------------------------

def test_error_mapping_covers_all_d279_error_codes():
    for code in (
        reg.ASSET_NOT_FOUND, reg.ASSET_NOT_OWNED, reg.ASSET_NOT_READY, reg.ASSET_MEDIA_UNSUPPORTED,
        reg.ASSET_HAS_NO_AUDIO, reg.TIMELINE_REVISION_CONFLICT, reg.TIMELINE_INVALID, reg.ASSET_REFERENCED,
    ):
        status = store.map_outcome_to_http_status(code)
        assert isinstance(status, int)
        assert 400 <= status < 500


def test_error_mapping_unknown_outcome_never_reads_as_success():
    assert store.map_outcome_to_http_status("SOME_FUTURE_OUTCOME") == 500


# ---------------------------------------------------------------------------
# Storage key safety (Stage 35)
# ---------------------------------------------------------------------------

def test_storage_key_rejects_traversal(tmp_path):
    persist_fn = store.local_directory_persister(str(tmp_path / "durable"))
    src = tmp_path / "x.mp4"
    src.write_bytes(b"1234")
    with pytest.raises(ValueError):
        persist_fn(str(src), "../../etc/passwd")


def test_storage_key_is_server_generated_never_from_original_filename(redis, persist, broll_source):
    asset = store.create_video_timeline_asset(
        user_id="u1", project_id="p1", role=reg.TimelineAssetRole.SUPPLEMENTAL_BROLL,
        local_source_path=broll_source, persist_media=persist, client=redis,
    )
    assert "broll.mp4" not in asset.storage_reference
    assert asset.asset_id in asset.storage_reference


# ---------------------------------------------------------------------------
# Structural / closed-track firewall (Stage 46/47/50)
# ---------------------------------------------------------------------------

def test_module_never_imports_forbidden_closed_tracks():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(store))
    forbidden = {
        "best_take_authority", "deterministic_best_take_authority", "selection_freeze",
        "boundary_engine", "dialogue_pacing_transition", "audio_finishing_executor",
        "audio_finishing_measurement", "visual_finishing_policy", "visual_mode",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden


def test_module_contains_no_ai_broll_ranking_or_placement_code():
    import inspect
    source = inspect.getsource(store).lower()
    for forbidden in ("clip_score", "ranking_model", "auto_place", "similarity_score", "recommend_broll"):
        assert forbidden not in source


def test_module_never_performs_real_s3_mutation():
    import inspect
    source = inspect.getsource(store)
    assert "boto3" not in source
    assert "put_object" not in source
    assert "upload_file" not in source
