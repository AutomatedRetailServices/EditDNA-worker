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


def test_export_submit_does_not_wait_for_the_worker_wake(monkeypatch):
    import threading
    import time as _time
    from cutsell_worker import queueing

    class FakeJob:
        id = "job-1"

    class FakeQueue:
        name = "cutsell"

        def enqueue(self, *args, **kwargs):
            return FakeJob()

    woke = threading.Event()

    def slow_wake():
        _time.sleep(1.0)
        woke.set()

    monkeypatch.setattr(queueing, "wake_worker", slow_wake)
    started = _time.monotonic()
    submission = queueing.enqueue_export({"user_id": "u", "project_id": "p"}, queue=FakeQueue())
    assert submission.job_id == "job-1"
    assert _time.monotonic() - started < 0.5          # returned before the wake call finished
    assert woke.wait(3)                                # ...and the wake call still happens


def test_creator_placed_media_is_not_judged_as_an_engine_mistake(tmp_path):
    """A silent, still B-roll video from the phone and a hand-dragged clip end used to be trimmed
    by the export QC as "accidental silence / frozen picture" and the export failed."""
    from cutsell_worker.live_render_qc import render_with_post_render_qc, user_placed_windows
    from cutsell_worker.render_plan import build_render_plan
    from cutsell_worker.live_boundary_repair import segment_output_windows
    from cutsell_worker.serde import draft_from_dict

    main = tmp_path / "main.mp4"
    broll = tmp_path / "broll.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=360x640:r=30:d=8",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(main)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=360x640:r=30:d=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(broll)], check=True)
    payload = {
        "project_id": "p1", "captions_enabled": False,
        "selected": [
            {"clip_id": "a", "source_asset_id": "s1", "start": 0.0, "end": 2.0, "text": "hi"},
            {"clip_id": "broll_x", "source_asset_id": "src_broll_x", "source_order": 1, "start": 0.0, "end": 3.0,
             "text": "", "caption_text": "", "words": []},
            {"clip_id": "b", "source_asset_id": "s1", "start": 4.0, "end": 6.0, "text": "there"},
        ],
        "broll_sources": [{"source_asset_id": "src_broll_x", "uri": "s3://b/k/broll.mp4"}],
    }
    draft = draft_from_dict(payload)
    assert draft.user_media_source_ids == ("src_broll_x",)
    plan = build_render_plan(draft, {"s1": str(main), "src_broll_x": str(broll)})
    windows = user_placed_windows(draft, plan, segment_output_windows(plan))
    assert len(windows) == 1 and windows[0][1] - windows[0][0] == pytest.approx(3.0, abs=0.1)
    result = render_with_post_render_qc(draft, plan, str(tmp_path / "out.mp4"))
    assert result.status == "PASS"
    # Without the B-roll being marked as placed by the creator, the same video is "repaired" away.
    plain = draft_from_dict({k: v for k, v in payload.items() if k != "broll_sources"})
    unmarked = render_with_post_render_qc(plain, build_render_plan(plain, {"s1": str(main), "src_broll_x": str(broll)}),
                                          str(tmp_path / "out2.mp4"))
    assert unmarked.status != "PASS" or len(unmarked.attempts) > 1


def test_hand_dragged_end_tail_is_a_placed_window():
    from cutsell_worker.live_render_qc import user_placed_windows
    from cutsell_worker.render_plan import build_render_plan
    from cutsell_worker.live_boundary_repair import segment_output_windows
    from cutsell_worker.serde import draft_from_dict
    draft = draft_from_dict({
        "project_id": "p1", "captions_enabled": False,
        "selected": [{"clip_id": "a", "source_asset_id": "s1", "start": 1.0, "end": 5.0, "text": "hi", "manual_end": True,
                      "words": [{"text": "hi", "start": 1.2, "end": 2.0}]}],
    })
    plan = build_render_plan(draft, {"s1": "/a.mp4"})
    assert user_placed_windows(draft, plan, segment_output_windows(plan)) == ((1.0, 4.0),)


def test_outside_windows_keeps_only_unplaced_parts():
    from cutsell_worker.post_render_media_qc import _outside_windows
    from cutsell_worker.post_render_watch_listen_qc import PostRenderFinding
    finding = PostRenderFinding(kind="FROZEN_OR_REPEATED_FRAME", start=2.0, end=8.0, detail={}, routes_to="BoundaryEngine")
    assert _outside_windows((finding,), [(2.0, 8.0)]) == ()
    left = _outside_windows((finding,), [(4.0, 9.0)])
    assert len(left) == 1 and left[0].start == 2.0 and left[0].end == pytest.approx(3.95)
