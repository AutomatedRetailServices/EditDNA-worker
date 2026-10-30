"""Voice isolation for "Clean up voice" (PO 30 Oct 2026): keep the person's voice, drop the music and the
room noise around it, the way CapCut's "Isolate voice" does -- instead of the old ffmpeg afftdn filter,
which left the music in and made the voice sound "like inside a fish tank".

Model: MDX-Net "Kim_Vocal_2" (ONNX, 66.8 MB, vocals stem), run with onnxruntime on the CPU. Only numpy +
onnxruntime are needed (no torch): the STFT / ISTFT below match torch.stft / torch.istft (centered, periodic
Hann window), and the chunking matches the UVR / python-audio-separator MDX inference
(hop 1024, segment 256 frames, 25 % overlap with a Hann cross-fade, first 3 bins zeroed).

The model file is baked into the worker image (modal_cutsell_worker.py) and checked by its sha256; when it
is missing or fails, `isolate_voice` raises -- the caller records which cleaning was used, never silently.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

MODEL_NAME = "Kim_Vocal_2.onnx"
MODEL_SHA256 = "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b"
MODEL_URL = ("https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/" + MODEL_NAME)
MODEL_DIR_ENV = "CUTSELL_VOICE_MODEL_DIR"
DEFAULT_MODEL_DIR = "/root/models"

SAMPLE_RATE = 44_100
N_FFT = 7680
HOP = 1024
DIM_F = 3072
SEGMENT = 256                      # frames per chunk (= the model's dim_t, so onnxruntime runs it as is)
OVERLAP = 0.25
TRIM = N_FFT // 2
CHUNK = HOP * (SEGMENT - 1)        # 261 120 samples (~5.9 s)
PEAK = 0.9

_session = None


def model_path() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR) / MODEL_NAME


def model_ok(path: Optional[Path] = None) -> bool:
    path = path or model_path()
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == MODEL_SHA256


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        path = model_path()
        if not model_ok(path):
            raise RuntimeError(f"voice model missing or wrong checksum: {path}")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        _session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    return _session


_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic Hann


def _stft(x: np.ndarray) -> np.ndarray:
    """x (2, n) -> model input (1, 4, DIM_F, frames): [L re, L im, R re, R im], like torch.stft(center=True)."""
    padded = np.pad(x, ((0, 0), (TRIM, TRIM)), mode="reflect")
    frames = 1 + (padded.shape[-1] - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    spec = np.fft.rfft(padded[:, index] * _WINDOW, axis=-1)          # (2, frames, bins)
    spec = spec[:, :, :DIM_F].transpose(0, 2, 1)                      # (2, DIM_F, frames)
    out = np.empty((1, 4, DIM_F, frames), dtype=np.float32)
    out[0, 0], out[0, 1] = spec[0].real, spec[0].imag
    out[0, 2], out[0, 3] = spec[1].real, spec[1].imag
    return out


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    """(1, 4, DIM_F, frames) -> (2, length), like torch.istft(center=True)."""
    bins = N_FFT // 2 + 1
    frames = spec.shape[-1]
    out = np.zeros((2, HOP * (frames - 1) + N_FFT), dtype=np.float64)
    norm = np.zeros(out.shape[-1], dtype=np.float64)
    for channel in range(2):
        full = np.zeros((bins, frames), dtype=np.complex64)
        full[:DIM_F] = spec[0, 2 * channel] + 1j * spec[0, 2 * channel + 1]
        wave = np.fft.irfft(full.T, n=N_FFT, axis=-1) * _WINDOW          # (frames, N_FFT)
        for f in range(frames):
            out[channel, f * HOP:f * HOP + N_FFT] += wave[f]
    for f in range(frames):
        norm[f * HOP:f * HOP + N_FFT] += _WINDOW.astype(np.float64) ** 2
    out /= np.where(norm > 1e-11, norm, 1.0)
    return out[:, TRIM:TRIM + length].astype(np.float32)


def _run_chunk(session, chunk: np.ndarray) -> np.ndarray:
    spec = _stft(chunk)
    spec[:, :, :3, :] = 0.0
    pred = session.run(None, {"input": spec})[0]
    return _istft(pred, CHUNK)


def isolate_voice(mix: np.ndarray, progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
    """mix: float32 (2, n) at 44.1 kHz -> the voice alone, same shape and level."""
    if mix.ndim != 2 or mix.shape[0] != 2 or mix.shape[1] == 0:
        raise ValueError("isolate_voice expects stereo audio shaped (2, n)")
    session = _get_session()
    peak = float(np.abs(mix).max())
    scale = PEAK / peak if peak > PEAK else 1.0
    mix = mix.astype(np.float32) * scale

    gen = CHUNK - 2 * TRIM
    pad = gen + TRIM - (mix.shape[-1] % gen)
    mixture = np.concatenate([np.zeros((2, TRIM), np.float32), mix, np.zeros((2, pad), np.float32)], axis=1)
    step = int((1 - OVERLAP) * CHUNK)
    result = np.zeros_like(mixture)
    divider = np.zeros_like(mixture)
    starts = list(range(0, mixture.shape[-1], step))
    for done, start in enumerate(starts, 1):
        end = min(start + CHUNK, mixture.shape[-1])
        part = mixture[:, start:end]
        if part.shape[-1] < CHUNK:
            part = np.concatenate([part, np.zeros((2, CHUNK - part.shape[-1]), np.float32)], axis=1)
        window = np.hanning(end - start).astype(np.float32)[None, :]
        voice = _run_chunk(session, part)[:, :end - start] * window
        result[:, start:end] += voice
        divider[:, start:end] += window
        if progress:
            progress(done / len(starts))
    voice = (result / np.where(divider > 0, divider, 1.0))[:, TRIM:TRIM + mix.shape[-1]]
    return (voice / scale).astype(np.float32)


def isolate_file(input_path: str, output_wav: str, progress: Optional[Callable[[float], None]] = None) -> float:
    """Any audio / video file -> a 44.1 kHz stereo float WAV of the voice alone. Returns the seconds of sound."""
    import subprocess

    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(input_path), "-vn", "-ac", "2", "-ar",
         str(SAMPLE_RATE), "-f", "f32le", "-"], capture_output=True, check=True, timeout=1200).stdout
    mix = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    voice = isolate_voice(mix, progress=progress)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2",
         "-i", "-", "-c:a", "pcm_f32le", str(output_wav)],
        input=np.ascontiguousarray(voice.T).tobytes(), check=True, timeout=1200)
    return mix.shape[-1] / SAMPLE_RATE
