"""Editor v5 export wiring: the draft's audio lane is resolved to local files -- voice/music only
through the tenant-safe timeline asset registry, sounds from the official library in the media
bucket -- and handed to the render as one AudioMix. An unresolvable clip fails the export with
`audio_clip_unavailable`; nothing is silently dropped."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import cutsell_worker.export_job as export_job
from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker.audio_mix_render import AudioMix, LocalAudioClip
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker.serde import draft_from_dict


def _payload(**extra):
    draft = {
        "project_id": "project-1",
        "captions_enabled": False,
        "selected": [{"clip_id": "clip-1", "source_asset_id": "src-1", "start": 1.0, "end": 5.0, "text": "hello"}],
    }
    draft.update(extra)
    return draft


VOICE = {"audio_id": "a1", "kind": "voice", "asset_id": "asset-voice", "start": 0.5, "source_in": 0.2,
         "source_out": 1.7, "volume": 1.2, "duck_video": True, "clean": True}
MUSIC = {"audio_id": "a2", "kind": "music", "asset_id": "asset-music", "start": 0.0, "source_in": 0.0,
         "source_out": 30.0, "volume": 0.4}
TAP = {"audio_id": "a3", "kind": "sfx", "sfx_id": "tap", "start": 2.0, "source_in": 0.0, "source_out": 0.3}
TAP2 = dict(TAP, audio_id="a4", start=3.0)


@pytest.fixture
def registry(monkeypatch):
    calls = []

    def fake_resolve(*, user_id, project_id, asset_id, required_media_kind, materialize=None, **kwargs):
        calls.append((user_id, project_id, asset_id, required_media_kind))
        if asset_id.startswith("missing"):
            return reg.AssetResolutionResult(outcome=reg.ASSET_NOT_FOUND, reason_codes=(reg.ASSET_NOT_FOUND,))
        if asset_id.startswith("foreign"):
            return reg.AssetResolutionResult(outcome=reg.ASSET_NOT_OWNED, reason_codes=(reg.ASSET_NOT_OWNED,))
        from cutsell_worker.timeline_composition_executor import ResolvedTimelineAsset
        return reg.AssetResolutionResult(outcome=reg.ASSET_RESOLVED, resolved=ResolvedTimelineAsset(
            asset_id=asset_id, local_path=f"/job/{asset_id}.m4a", duration_sec=30.0, has_audio=True))

    monkeypatch.setattr(export_job.timeline_store, "resolve_timeline_asset_live", fake_resolve)
    return calls


@pytest.fixture
def sfx_downloads(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "cutsell-media-test")
    downloads = []

    def fake_download(uri, destination):
        downloads.append((uri, destination))
        return destination

    monkeypatch.setattr(export_job, "download_source", fake_download)
    return downloads


def test_no_audio_fields_means_no_mix(registry, sfx_downloads, tmp_path):
    draft = draft_from_dict(_payload())
    assert export_job.resolve_audio_mix(draft, user_id="u", project_id="project-1", directory=str(tmp_path)) is None
    assert registry == [] and sfx_downloads == []


def test_audio_lane_resolves_through_registry_and_library(registry, sfx_downloads, tmp_path):
    draft = draft_from_dict(_payload(audio_clips=[VOICE, MUSIC, TAP, TAP2], video_volume=0.6, video_clean_voice=True))
    mix = export_job.resolve_audio_mix(draft, user_id="u", project_id="project-1", directory=str(tmp_path))
    assert registry == [("u", "project-1", "asset-voice", reg.TimelineMediaKind.AUDIO),
                        ("u", "project-1", "asset-music", reg.TimelineMediaKind.AUDIO)]
    # The same sound used twice is downloaded once, from the official library key.
    assert [uri for uri, _ in sfx_downloads] == ["s3://cutsell-media-test/assets/sfx/v1/tap_select.m4a"]
    assert sfx_downloads[0][1].startswith(str(tmp_path))
    assert mix.video_volume == 0.6 and mix.video_clean is True
    assert mix.clips[0] == LocalAudioClip("/job/asset-voice.m4a", 0.5, 0.2, 1.7, volume=1.2, duck_video=True, clean=True)
    assert mix.clips[1] == LocalAudioClip("/job/asset-music.m4a", 0.0, 0.0, 30.0, volume=0.4)
    assert mix.clips[2].path == mix.clips[3].path == sfx_downloads[0][1]
    assert (mix.clips[3].start, mix.clips[3].source_out) == (3.0, 0.3)


def test_volume_only_mix(registry, sfx_downloads, tmp_path):
    mix = export_job.resolve_audio_mix(draft_from_dict(_payload(video_volume=0)), user_id="u",
                                       project_id="project-1", directory=str(tmp_path))
    assert mix == AudioMix(video_volume=0.0) and not mix.is_noop


@pytest.mark.parametrize("asset_id,reason", [("missing-1", reg.ASSET_NOT_FOUND), ("foreign-1", reg.ASSET_NOT_OWNED)])
def test_unavailable_voice_fails_with_a_clear_code(registry, sfx_downloads, tmp_path, asset_id, reason):
    draft = draft_from_dict(_payload(audio_clips=[dict(VOICE, asset_id=asset_id)]))
    with pytest.raises(export_job.AudioClipUnavailable) as caught:
        export_job.resolve_audio_mix(draft, user_id="u", project_id="project-1", directory=str(tmp_path))
    assert caught.value.error_code == "audio_clip_unavailable"
    assert str(caught.value) == f"audio_clip_unavailable:a1:{reason}"


def test_sfx_download_failure_is_not_silent(monkeypatch, tmp_path):
    monkeypatch.setenv("S3_BUCKET", "cutsell-media-test")

    def broken(uri, destination):
        raise OSError("no such key")

    monkeypatch.setattr(export_job, "download_source", broken)
    with pytest.raises(export_job.AudioClipUnavailable, match="audio_clip_unavailable:a3:OSError"):
        export_job.resolve_audio_mix(draft_from_dict(_payload(audio_clips=[TAP])), user_id="u",
                                     project_id="project-1", directory=str(tmp_path))


def test_registry_is_called_with_a_project_scoped_s3_materializer(monkeypatch, tmp_path):
    seen = {}

    def fake_resolve(**kwargs):
        seen.update(kwargs)
        return reg.AssetResolutionResult(outcome=reg.ASSET_NOT_READY, reason_codes=(reg.ASSET_NOT_READY,))

    monkeypatch.setattr(export_job.timeline_store, "resolve_timeline_asset_live", fake_resolve)
    with pytest.raises(export_job.AudioClipUnavailable):
        export_job.resolve_audio_mix(draft_from_dict(_payload(audio_clips=[VOICE])), user_id="u",
                                     project_id="project-1", directory=str(tmp_path))
    assert seen["user_id"] == "u" and seen["project_id"] == "project-1" and seen["asset_id"] == "asset-voice"
    assert callable(seen["materialize"])
    # The materializer refuses anything outside this project's own storage scope.
    monkeypatch.setenv("S3_BUCKET", "cutsell-media-test")
    with pytest.raises(RuntimeError):
        seen["materialize"]("s3://cutsell-media-test/cutsell/uploads/other-user/other-project/x.m4a")


# --- run_export_job ------------------------------------------------------------------------------

class FakeJob:
    def __init__(self):
        self.id = "job-v5-audio"
        self.started_at = 1_700_000_000.0
        self.meta = {}

    def save_meta(self):
        pass


@pytest.fixture
def export_env(monkeypatch, registry, sfx_downloads):
    rq_module = ModuleType("rq")
    rq_module.get_current_job = lambda: FakeJob()
    monkeypatch.setitem(sys.modules, "rq", rq_module)
    monkeypatch.setattr(export_job, "validate_product_source_uri", lambda uri, **kw: ("bucket", "key"))
    original_download = export_job.download_source

    def download(uri, destination):
        if "assets/sfx/" in uri:
            return original_download(uri, destination)
        Path(destination).write_bytes(b"source")
        return destination

    monkeypatch.setattr(export_job, "download_source", download)
    plan = (RenderSegment(clip_id="clip-1", source_asset_id="src-1", source_path="/tmp/x.mp4", start=1.0, end=5.0),)
    monkeypatch.setattr(export_job, "build_render_plan", lambda draft, local_paths: plan)
    notes = []
    monkeypatch.setattr(export_job, "_safe_notify", lambda **kw: notes.append(kw) or {})
    monkeypatch.setattr(export_job, "safe_update_project", lambda **kw: {})
    rendered = []

    def fake_render(draft, plan, output, **kwargs):
        rendered.append(kwargs)
        raise RuntimeError("stop after render call")

    monkeypatch.setattr(export_job, "render_with_post_render_qc", fake_render)
    return rendered, notes


def _run(draft):
    return export_job.run_export_job({
        "project_id": "project-1", "user_id": "user-1", "draft": draft,
        "sources": [{"source_asset_id": "src-1", "original_name": "one.mov", "uri": "s3://bucket/cutsell/uploads/u/p/one.mov"}],
    })


def test_export_passes_the_mix_to_the_render(export_env):
    rendered, _notes = export_env
    with pytest.raises(RuntimeError, match="stop after render"):
        _run(_payload(audio_clips=[VOICE, TAP], video_volume=0.8))
    mix = rendered[0]["audio_mix"]
    assert isinstance(mix, AudioMix) and mix.video_volume == 0.8 and len(mix.clips) == 2


def test_export_without_audio_keeps_the_old_render_call(export_env):
    rendered, _notes = export_env
    with pytest.raises(RuntimeError):
        _run(_payload())
    assert "audio_mix" not in rendered[0]


def test_export_records_download_audio_timing(export_env, monkeypatch):
    def fake_render(draft, plan, output, **kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(export_job, "render_with_post_render_qc", fake_render)
    job = FakeJob()
    sys.modules["rq"].get_current_job = lambda: job
    with pytest.raises(RuntimeError):
        _run(_payload(audio_clips=[TAP]))
    timings = job.meta.get("timings") or {}
    assert list(timings)[:3] == ["download_sources", "download_overlays", "download_audio"]


def test_unavailable_clip_fails_the_export_with_its_code(export_env):
    rendered, notes = export_env
    with pytest.raises(export_job.AudioClipUnavailable):
        _run(_payload(audio_clips=[dict(VOICE, asset_id="foreign-asset")]))
    assert rendered == []  # never rendered without the clip
    failed = [n for n in notes if n.get("kind") == "render_failed"]
    assert failed and failed[-1]["payload"]["error_code"] == "audio_clip_unavailable"
    assert failed[-1]["payload"]["error"] == "AudioClipUnavailable"
