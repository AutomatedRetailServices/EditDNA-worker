from array import array
from pathlib import Path

import cutsell_worker.timeline_assets as assets


def test_filmstrip_times_are_bounded_and_cover_source():
    times = assets.filmstrip_times(10.0, max_frames=6)
    assert len(times) == 6
    assert times[0] == 0.0
    assert 9.9 < times[-1] < 10.0
    assert list(times) == sorted(times)


def test_generate_filmstrip_uses_real_timestamps_without_modifying_source(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    calls = []

    def fake_run(command, capture=False):
        calls.append(command)
        Path(command[-1]).write_bytes(b"jpg")
        return b""

    monkeypatch.setattr(assets, "_run", fake_run)
    result = assets.generate_filmstrip(str(source), str(tmp_path / "frames"), duration_sec=4.0, max_frames=3)
    assert len(result) == 3
    assert all(Path(item["path"]).exists() for item in result)
    assert source.read_bytes() == b"video"
    assert all("-ss" in command and "-frames:v" in command for command in calls)


def test_waveform_peaks_are_normalized_and_fixed_length(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    pcm = array("h", [0, 1000, -4000, 8000, -16000, 32767, 0, 0]).tobytes()
    monkeypatch.setattr(assets, "_run", lambda command, capture=False: pcm)
    peaks = assets.waveform_peaks(str(source), buckets=32)
    assert len(peaks) == 32
    assert all(0.0 <= peak <= 1.0 for peak in peaks)
    assert max(peaks) > 0.9
