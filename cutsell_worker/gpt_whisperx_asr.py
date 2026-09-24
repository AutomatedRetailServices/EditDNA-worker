"""Opt-in RAW ASR: OpenAI text, isolated WhisperX forced alignment.

This supplies evidence to the existing engine; it is not an editor. No
Whisper decode, guessed timestamps, provider fallback, or transcript repair.
The heavy aligner runs in a separate Python environment to preserve the
clean worker's Faster-Whisper / Torch / MediaPipe dependency boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import requests

from .audio_silence import detect_audio_silence_intervals
from .contracts import TranscriptSegment, Word

PROVIDER = "gpt-transcribe-whisperx"
PROVIDER_ENV = "CUTSELL_VALIDATION_ASR_PROVIDER"
ALIGNER_PYTHON = "/opt/cutsell-whisperx/bin/python"
ALIGNER_SCRIPT = "/opt/cutsell-whisperx/align.py"
OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
TARGET_CHUNK_SEC = 30.0
MAX_CHUNK_SEC = 40.0


class AlignmentEvidenceError(RuntimeError):
    """No usable word timing evidence: never silently revert to another ASR."""


@dataclass(frozen=True)
class ProviderFingerprint:
    language_hint: str | None = None

    def fingerprint(self) -> str:
        spec = {"provider": PROVIDER, "whisperx": "3.8.6", "model": "gpt-transcribe",
                "alignment": "torchaudio-default-en-es", "interpolation": "ignore",
                "chunk_target": TARGET_CHUNK_SEC, "chunk_max": MAX_CHUNK_SEC,
                "silence_db": -35, "silence_sec": 0.6, "prompt": None,
                "audio": "pcm_s16le-mono-16000", "policy": "strict-word-coverage-v1",
                "language_hint": self.language_hint}
        return "asrcfg_" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def chunk_windows(duration: float, silences: tuple) -> list[tuple[float, float]]:
    """Contiguous complete coverage; prefer real quiet midpoints, bound memory.

    A source without nearby silence gets a disclosed hard boundary. There
    is no overlap or deduplication that could remove a genuine retake.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("ASR requires a finite positive audio duration")
    mids = sorted((float(a) + float(b)) / 2 for a, b in silences
                  if 0 <= a < b <= duration and b - a >= 0.6)
    windows = []
    start = 0.0
    while duration - start > MAX_CHUNK_SEC:
        candidates = [p for p in mids if start + 20 <= p <= start + MAX_CHUNK_SEC]
        end = min(candidates, key=lambda p: (abs(p - start - TARGET_CHUNK_SEC), p)) if candidates else start + TARGET_CHUNK_SEC
        windows.append((start, end))
        start = end
    windows.append((start, duration))
    return windows


def checked_segments(chunk: dict, aligned: dict, source_asset_id: str) -> tuple[TranscriptSegment, ...]:
    """Retain every provider token once, in order, with observed positive times."""
    expected = chunk["text"].split()
    raw_segments = aligned.get("segments", [])
    raw_words = [w for segment in raw_segments for w in segment.get("words", [])]
    actual = [str(w.get("word", "")).strip() for w in raw_words]
    if expected != actual:
        raise AlignmentEvidenceError(f"chunk {chunk['index']}: alignment changed or omitted transcript words")
    output = []
    previous_end = 0.0
    duration = chunk["end"] - chunk["start"]
    for segment in raw_segments:
        words = []
        for raw in segment.get("words", []):
            try:
                start, end = float(raw["start"]), float(raw["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AlignmentEvidenceError(f"chunk {chunk['index']}: missing word timing") from exc
            if not all(map(math.isfinite, (start, end))) or not (0 <= start < end <= duration + 0.005) or start < previous_end - 0.002:
                raise AlignmentEvidenceError(f"chunk {chunk['index']}: invalid/overlapping word timing ({start}, {end})")
            score = raw.get("score")
            if score is not None and (not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
                raise AlignmentEvidenceError(f"chunk {chunk['index']}: invalid alignment score")
            words.append(Word(str(raw["word"]).strip(), chunk["start"] + start,
                              chunk["start"] + end, float(score) if score is not None else None))
            previous_end = end
        if words:
            output.append(TranscriptSegment(source_asset_id, words[0].start, words[-1].end,
                                            " ".join(w.text for w in words), tuple(words)))
    return tuple(output)


@dataclass
class GPTWhisperXASR:
    model_name: str = "gpt-transcribe+whisperx-3.8.6"
    last_audit: dict = field(default_factory=dict, init=False)

    def config_fingerprint(self, *, language_hint: str | None = None) -> ProviderFingerprint:
        return ProviderFingerprint(language_hint)

    def transcribe(self, path: str, *, source_asset_id: str, language_hint: str | None = None) -> tuple[TranscriptSegment, ...]:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key or key.startswith("sk-admin-"):
            raise RuntimeError("GPT/WhisperX requires an ordinary OPENAI_API_KEY in the worker environment")
        if not Path(ALIGNER_PYTHON).is_file() or not Path(ALIGNER_SCRIPT).is_file():
            raise RuntimeError("The isolated WhisperX environment is missing")
        if language_hint is not None and language_hint not in {"en", "es"}:
            raise ValueError("This qualified alignment image supports en/es only")
        began = time.monotonic()
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                               check=True, capture_output=True, text=True, timeout=60)
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        silence = detect_audio_silence_intervals(path, noise_db=-35, minimum_silence_sec=0.6,
                                                 probe_minimum_sec=0.6, merge_gap_sec=0)
        windows = chunk_windows(duration, silence)
        self.last_audit = {"provider": PROVIDER, "model": "gpt-transcribe", "duration_sec": duration,
                           "config_fingerprint": self.config_fingerprint(language_hint=language_hint).fingerprint(),
                           "chunks": [], "status": "running", "timestamp_interpolation": False,
                           "fallback": None, "prompt": None, "requested_language": language_hint}
        try:
            with tempfile.TemporaryDirectory(prefix="cutsell-gpt-whisperx-") as work:
                work = Path(work)
                for index, (start, end) in enumerate(windows):
                    audio = work / f"chunk-{index:04d}.wav"
                    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-ss", str(start),
                                    "-t", str(end - start), "-vn", "-ar", "16000", "-ac", "1",
                                    "-c:a", "pcm_s16le", "-y", str(audio)], check=True, capture_output=True, timeout=120)
                    fields = [("model", "gpt-transcribe")]
                    if language_hint:
                        fields.append(("languages[]", language_hint))
                    requested = time.monotonic()
                    with audio.open("rb") as fh:
                        response = requests.post(OPENAI_TRANSCRIPTIONS_URL,
                                                 headers={"Authorization": f"Bearer {key}"}, data=fields,
                                                 files={"file": (audio.name, fh, "audio/wav")}, timeout=(15, 120))
                    if response.status_code != 200:
                        raise RuntimeError(f"OpenAI transcription HTTP {response.status_code}; request_id={response.headers.get('x-request-id', 'unknown')}")
                    payload = response.json()
                    text = payload.get("text")
                    if not isinstance(text, str):
                        raise RuntimeError("OpenAI transcription returned no text field")
                    languages = [x.get("code") for x in payload.get("languages", []) if isinstance(x, dict)]
                    language = language_hint or next((x for x in languages if x in {"en", "es"}), None)
                    if text.strip() and (language is None or any(x not in {"en", "es"} for x in languages)):
                        raise AlignmentEvidenceError(f"chunk {index}: no qualified alignment language")
                    quiet_boundary = index == len(windows) - 1 or any(a <= end <= b for a, b in silence)
                    chunk = {"index": index, "start": start, "end": end, "text": " ".join(text.split()),
                             "raw_provider_text": text,
                             "language": language, "detected_languages": languages,
                             "end_boundary": "silence_or_eof" if quiet_boundary else "hard_window",
                             "request_id": response.headers.get("x-request-id"),
                             "gpt_elapsed_sec": round(time.monotonic() - requested, 3)}
                    self.last_audit["chunks"].append(chunk)
                    print(f"GPT/WhisperX: text chunk {index + 1}/{len(windows)}, words={len(text.split())}", flush=True)
                manifest = {"chunks": self.last_audit["chunks"]}
                (work / "input.json").write_text(json.dumps(manifest), encoding="utf-8")
                # Aligner needs no cloud credentials. Keep only runtime/cache/GPU controls.
                align_env = {k: v for k, v in os.environ.items() if k in {
                    "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "HF_HOME", "TORCH_HOME",
                    "NLTK_DATA", "TMPDIR"}}
                proc = subprocess.run([ALIGNER_PYTHON, ALIGNER_SCRIPT, str(work / "input.json"),
                                       str(work / "output.json")], env=align_env, capture_output=True,
                                      text=True, timeout=900)
                if proc.returncode:
                    # No credentials enter the sidecar; retain bounded diagnostics on failure.
                    raise AlignmentEvidenceError("WhisperX sidecar failed: " + proc.stderr[-4000:])
                aligned = json.loads((work / "output.json").read_text(encoding="utf-8"))
                if len(aligned["chunks"]) != len(windows):
                    raise AlignmentEvidenceError("WhisperX changed chunk count")
                self.last_audit["alignment_runtime"] = aligned["runtime"]
                output = []
                for chunk, result in zip(self.last_audit["chunks"], aligned["chunks"]):
                    if result["index"] != chunk["index"]:
                        raise AlignmentEvidenceError("WhisperX changed chunk order")
                    segments = checked_segments(chunk, result, source_asset_id)
                    chunk["alignment_model"] = result.get("alignment_model")
                    chunk["aligned_word_count"] = sum(len(s.words) for s in segments)
                    output.extend(segments)
                self.last_audit.update(status="passed", word_count=sum(len(s.words) for s in output),
                                       segment_count=len(output), elapsed_sec=round(time.monotonic() - began, 3))
                return tuple(output)
        except Exception as exc:
            self.last_audit.update(status="failed", error=str(exc), elapsed_sec=round(time.monotonic() - began, 3))
            # The ordinary job fails closed; preserve evidence in the protected workflow log.
            print("GPT_WHISPERX_ASR_AUDIT=" + json.dumps(self.last_audit, ensure_ascii=False), flush=True)
            raise
