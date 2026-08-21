from types import SimpleNamespace

from cutsell_worker.render import _tighten_trailing_silence
from cutsell_worker.render_plan import RenderSegment


def _segment():
    return RenderSegment(
        clip_id="clip",
        source_asset_id="src",
        source_path="/tmp/source.mp4",
        start=10.0,
        end=15.0,
    )


def test_trailing_silence_reaching_edge_is_removed(monkeypatch):
    monkeypatch.setattr("cutsell_worker.render.probe_media", lambda _path: SimpleNamespace(has_audio=True))
    completed = SimpleNamespace(
        returncode=0,
        stdout="",
        stderr="[silencedetect] silence_start: 4.000\n[silencedetect] silence_end: 5.000 | silence_duration: 1.000\n",
    )
    monkeypatch.setattr("cutsell_worker.render.subprocess.run", lambda *args, **kwargs: completed)

    tightened = _tighten_trailing_silence(_segment())

    assert round(tightened.end, 2) == 14.04
    assert tightened.start == 10.0


def test_internal_silence_does_not_change_render_boundary(monkeypatch):
    monkeypatch.setattr("cutsell_worker.render.probe_media", lambda _path: SimpleNamespace(has_audio=True))
    completed = SimpleNamespace(
        returncode=0,
        stdout="",
        stderr="[silencedetect] silence_start: 1.000\n[silencedetect] silence_end: 2.000 | silence_duration: 1.000\n",
    )
    monkeypatch.setattr("cutsell_worker.render.subprocess.run", lambda *args, **kwargs: completed)

    original = _segment()
    assert _tighten_trailing_silence(original) == original


def test_segment_without_audio_is_unchanged(monkeypatch):
    monkeypatch.setattr("cutsell_worker.render.probe_media", lambda _path: SimpleNamespace(has_audio=False))
    original = _segment()
    assert _tighten_trailing_silence(original) == original
