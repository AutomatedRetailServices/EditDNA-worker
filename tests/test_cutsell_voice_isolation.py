"""Clean up voice v2 (PO 30 Oct 2026): AI voice isolation instead of the afftdn filter.

No model file in CI: the numeric parts (STFT / ISTFT, chunking, levels) run with a stand-in session that
returns its input, and make_clean_audio is checked both with a stand-in isolation and without the model
(the v1 filter, recorded as method "filter" -- never silent)."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from cutsell_worker import audio_clean, voice_isolation as vi


class _Identity:
    """Stands in for the ONNX session: 'the voice' is the whole input spectrum."""

    def __init__(self):
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        assert feeds["input"].shape == (1, 4, vi.DIM_F, vi.SEGMENT)
        return [feeds["input"]]


def test_stft_and_istft_give_the_sound_back():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, vi.CHUNK)).astype(np.float32) * 0.1
    spec = vi._stft(x)
    assert spec.shape == (1, 4, vi.DIM_F, vi.SEGMENT)
    # Only the bins the model sees are kept (up to 3072 of 3841, ~17.6 kHz): a low tone comes back as it was.
    t = np.arange(vi.CHUNK) / vi.SAMPLE_RATE
    tone = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]).astype(np.float32) * 0.3
    back = vi._istft(vi._stft(tone), vi.CHUNK)
    assert np.max(np.abs(back[:, 4000:-4000] - tone[:, 4000:-4000])) < 1e-3


def test_isolate_voice_keeps_length_level_and_joins_chunks_without_seams(monkeypatch):
    session = _Identity()
    monkeypatch.setattr(vi, "_get_session", lambda: session)
    n = int(14.3 * vi.SAMPLE_RATE)                      # several chunks with 25 % overlap
    t = np.arange(n) / vi.SAMPLE_RATE
    mix = np.stack([np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 500 * t)]).astype(np.float32) * 1.4  # > peak
    out = vi.isolate_voice(mix)
    assert out.shape == mix.shape and session.calls >= 3
    core = slice(8000, n - 8000)
    assert np.max(np.abs(out[:, core] - mix[:, core])) < 5e-3   # same sound, same level (peak scaling undone)


def test_isolate_voice_refuses_mono_and_a_missing_model(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        vi.isolate_voice(np.zeros((1, 100), np.float32))
    monkeypatch.setenv(vi.MODEL_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(vi, "_session", None)
    assert not vi.model_ok()
    with pytest.raises(RuntimeError, match="voice model missing"):
        vi.isolate_voice(np.zeros((2, 100), np.float32))


ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _voice_and_music(path, seconds=6.0):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"aevalsrc='0.3*sin(2*PI*220*t)*between(mod(t,2),0.2,1.2)|0.3*sin(2*PI*220*t)*between(mod(t,2),0.2,1.2)':d={seconds}:s=44100",
        "-f", "lavfi", "-i", f"aevalsrc='0.08*sin(2*PI*1500*t)|0.08*sin(2*PI*1500*t)':d={seconds}:s=44100",
        "-filter_complex", "amix=inputs=2:normalize=0", "-c:a", "aac", str(path)], check=True, capture_output=True)
    return str(path)


@ffmpeg
def test_clean_copy_uses_the_voice_model_and_no_fish_tank_filter(tmp_path, monkeypatch):
    src = _voice_and_music(tmp_path / "mix.m4a")
    seen = {}

    def fake_isolate(input_path, output_wav, progress=None):
        # "The voice alone": the 220 Hz part, the 1.5 kHz "music" removed.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", input_path, "-af", "lowpass=f=600,lowpass=f=600",
                        "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le", output_wav], check=True)
        if progress:
            progress(1.0)
        seen["called"] = True
        return 6.0

    monkeypatch.setattr(vi, "isolate_file", fake_isolate)
    chains = []
    real_run = subprocess.run

    def spy(cmd, *a, **k):
        if "-af" in cmd:
            chains.append(cmd[cmd.index("-af") + 1])
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(audio_clean.subprocess, "run", spy)
    out = str(tmp_path / "clean.m4a")
    facts = audio_clean.make_clean_audio(src, out)
    assert seen["called"] and facts["method"] == "voice_ai" and facts["model_error"] is None
    assert chains and not any("afftdn" in c or "agate" in c for c in chains)   # no "fish tank" filter after the AI
    assert facts["duration_sec"] == pytest.approx(6.0, abs=0.06)
    assert not list(tmp_path.glob("*.voice.wav"))                             # the work file is removed


@ffmpeg
def test_without_the_model_the_old_filter_runs_and_says_so(tmp_path, monkeypatch):
    src = _voice_and_music(tmp_path / "mix.m4a")
    monkeypatch.setenv(vi.MODEL_DIR_ENV, str(tmp_path / "none"))
    monkeypatch.setattr(vi, "_session", None)
    facts = audio_clean.make_clean_audio(src, str(tmp_path / "clean.m4a"))
    assert facts["method"] == "filter" and "voice model missing" in facts["model_error"]


def test_v2_copies_are_stored_apart_from_v1():
    key = audio_clean.clean_key(user_id="u1", project_id="p1", target="source", ident="s1")
    assert "/clean/v2/" in key
