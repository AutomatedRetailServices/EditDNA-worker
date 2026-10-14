"""Build 15 server side: export speed + real progress + stage timings, and real overlay / B-roll
links for the editor."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cutsell_worker import media_probe, post_render_media_qc
from cutsell_worker.export_job import _RenderProgressTicker

ROOT = Path(__file__).resolve().parents[1]


def test_progress_moves_from_35_to_84_and_never_back():
    expected = 20.0
    values = [_RenderProgressTicker.percent(t / 2, expected) for t in range(0, 200)]
    assert values[0] == 35
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert max(values) <= 84
    assert _RenderProgressTicker.percent(expected * 0.8, expected) == 80


def test_ticker_expected_time_grows_with_video_and_overlays():
    plain = _RenderProgressTicker(lambda *_: None, seconds_of_video=60)
    with_overlay = _RenderProgressTicker(lambda *_: None, seconds_of_video=60, media_overlays=1)
    short = _RenderProgressTicker(lambda *_: None, seconds_of_video=13)
    assert short.expected < plain.expected < with_overlay.expected


def test_silence_check_decodes_audio_only(monkeypatch):
    seen = {}

    def fake_run(command, **_):
        seen["command"] = command
        return 0, "silence_start: 1.0\nsilence_end: 2.0\n"

    monkeypatch.setattr(post_render_media_qc, "_run", fake_run)
    assert post_render_media_qc._detect_silence_intervals("x.mp4") == [(1.0, 2.0)]
    assert "-vn" in seen["command"]


def test_same_file_is_probed_once(tmp_path, monkeypatch):
    clip = tmp_path / "a.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.5:r=10",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    calls = {"n": 0}
    real = media_probe._probe_media_uncached

    def counting(path, runner):
        calls["n"] += 1
        return real(path, runner)

    monkeypatch.setattr(media_probe, "_probe_media_uncached", counting)
    media_probe._PROBE_CACHE.clear()
    first = media_probe.probe_media(str(clip))
    second = media_probe.probe_media(str(clip))
    assert first == second and calls["n"] == 1
    clip.write_bytes(clip.read_bytes() + b"\\0")          # a changed file is probed again
    media_probe.probe_media(str(clip))
    assert calls["n"] == 2


def test_export_reports_stage_timings_and_render_progress():
    source = (ROOT / "cutsell_worker" / "export_job.py").read_text()
    for stage in ("download_sources", "download_overlays", "plan", "render_and_qc", "upload_and_verify", "finalize"):
        assert f'mark("{stage}")' in source
    assert '"timings": dict(timings)' in source
    assert "ticker.start()" in source and "ticker.stop()" in source


def test_draft_read_signs_overlay_and_broll_links_and_save_strips_them(monkeypatch):
    pytest.importorskip("fastapi")
    from cutsell_app import main

    monkeypatch.setattr(main, "create_presigned_overlay_download",
                        lambda uri, **_: {"url": f"https://signed/{uri.rsplit('/', 1)[-1]}", "expires_in": 1800})
    monkeypatch.setattr(main, "create_presigned_source_download",
                        lambda uri, **_: {"url": f"https://signed/{uri.rsplit('/', 1)[-1]}", "expires_in": 1800})
    snapshot = {
        "project_id": "p1", "user_id": "u1", "sources": [],
        "draft": {"media_overlays": [{"overlay_id": "o1", "uri": "s3://b/k/photo.jpg"}],
                  "broll_sources": [{"source_asset_id": "src_b", "uri": "s3://b/k/broll.mp4"}]},
    }
    out = main._hydrate_snapshot_assets(snapshot)
    assert out["draft"]["media_overlays"][0]["preview_url"] == "https://signed/photo.jpg"
    assert out["draft"]["broll_sources"][0]["playback_url"] == "https://signed/broll.mp4"
    assert "preview_url" not in snapshot["draft"]["media_overlays"][0]          # the stored copy is untouched
    stripped = main._strip_signed_links(out["draft"])
    assert "preview_url" not in stripped["media_overlays"][0]
    assert "playback_url" not in stripped["broll_sources"][0]


def test_export_accepts_a_broll_clip_from_its_own_source():
    from cutsell_worker.render_plan import build_render_plan
    from cutsell_worker.serde import draft_from_dict
    draft = draft_from_dict({
        "project_id": "p1", "captions_enabled": False,
        "selected": [
            {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 1.0, "text": "hi"},
            {"clip_id": "broll_ab12", "source_asset_id": "src_broll_ab12", "source_order": 1, "start": 0.0,
             "end": 2.0, "text": "", "caption_text": "", "words": []},
        ],
        "broll_sources": [{"source_asset_id": "src_broll_ab12", "uri": "s3://b/k/broll.mp4"}],
    })
    plan = build_render_plan(draft, {"s1": "/a.mp4", "src_broll_ab12": "/b.mp4"})
    assert [seg.source_asset_id for seg in plan] == ["s1", "src_broll_ab12"]
