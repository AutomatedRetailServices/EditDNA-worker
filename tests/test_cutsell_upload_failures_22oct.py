"""22 Oct production failures: an iPhone video with a display rotation (+ VFR) blocked the cut at the format
gate, and a Deepgram call that never answered failed a cut after 600 s with no retry."""
import shutil
import subprocess

import pytest
import requests

from cutsell_worker.simple_engine import asr

ffmpeg = shutil.which("ffmpeg")


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation", "-of", "compact", path],
                         capture_output=True, text=True).stdout
    return out


@pytest.mark.skipif(not ffmpeg, reason="needs ffmpeg")
def test_iphone_rotated_vfr_video_passes_the_gate(tmp_path):
    from cutsell_worker.worker_job import resolve_sources_for_editorial_entry
    raw = tmp_path / "raw.mp4"
    src = tmp_path / "iphone.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=60000/1001,format=yuv420p",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "3",
                    "-vf", "select='not(eq(mod(n\\,7)\\,0))',setpts='PTS-STARTPTS'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-preset", "ultrafast", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-colorspace", "bt709", "-c:a", "aac", "-shortest", "-video_track_timescale", "600", str(raw)],
                   check=True)
    made = subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "-90", "-i", str(raw), "-c", "copy",
                           "-map", "0", str(src)])
    if made.returncode != 0:
        pytest.skip("this ffmpeg cannot write a display rotation")
    assert "rotation=-90" in _probe(str(src))
    resolved, _fmt, norm, blocked = resolve_sources_for_editorial_entry({"s1": str(src)}, output_directory=str(tmp_path))
    assert blocked == [], norm
    assert norm[0]["resolved_source_kind"] == "NORMALIZED"
    out = _probe(resolved["s1"])
    assert "width=360|height=640" in out and "rotation" not in out      # upright pixels, no matrix left


class _Resp:
    def __init__(self, code, payload=None):
        self.status_code = code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_deepgram_retries_a_timeout_then_succeeds(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\0" * 704000)                      # ~88 s at 64 kbps
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise requests.ReadTimeout("no answer")
        return _Resp(200, {"ok": True})

    assert asr._post_with_retries(str(audio), [], "k", post=post, sleep=lambda s: None) == {"ok": True}
    assert len(calls) == 2
    assert 180 <= calls[0][1] <= 200                       # ~60 + 1.5 x 88 s, not a flat 600 s


def test_deepgram_gives_up_after_three_and_does_not_retry_a_bad_request(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\0" * 1000)
    n = {"c": 0}

    def always_503(url, **kwargs):
        n["c"] += 1
        return _Resp(503)

    with pytest.raises(RuntimeError):
        asr._post_with_retries(str(audio), [], "k", post=always_503, sleep=lambda s: None)
    assert n["c"] == 3
    n["c"] = 0

    def bad(url, **kwargs):
        n["c"] += 1
        return _Resp(400)

    with pytest.raises(RuntimeError):
        asr._post_with_retries(str(audio), [], "k", post=bad, sleep=lambda s: None)
    assert n["c"] == 1
